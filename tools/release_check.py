"""Differential and isolation checks for a runtime release against the repository.

    run      score programs with one implementation, write normalised results
             --impl package|flat|release --root DIR (the import root)
    compile  translate + compile programs, write per-program hashes of the C text and
             of (memory, commands, checks)
    compare  A.json B.json: per-program equality; prints and writes the differences
    iso      fresh-process isolation + batch permutation check of a release

Results are normalised before comparing: latency and provenance fields dropped,
the C source replaced by its sha256, and the repository's message strings mapped
the same way the build maps them, so only execution can differ.
"""
import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VOLATILE = {'latency_s', 'runtime_version', 'runtime_diff_sha256', 'strandedcpu_commit',
            'strandedcpu_diff_sha256', 'loom_commit', 'loom_diff_sha256'}
TRUTH = {'true', 'false', 'yes', 'no'}


def load_impl(impl, root):
    sys.path.insert(0, root)
    if impl == 'package':
        from strandedcpu.python import score_python_batch, python_to_c
        from strandedcpu.compiler import compile_c
    elif impl == 'flat':
        from loom_py import score_python_batch, python_to_c
        from c_compiler import compile_c
    else:
        from xr.front import score_python_batch, python_to_c
        from xr.cc import compile_c
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None) or ''
        if name.split('.')[0] in ('strandedcpu', 'loom_py', 'c_compiler', 'xr', 'loom_api', 'isa_interpreter'):
            assert f.startswith(root), (name, f, root)
    return score_python_batch, python_to_c, compile_c


def mapper():
    sys.path.insert(0, HERE)
    import build_runtime as b
    sys.path.remove(HERE)

    def m(v):
        if isinstance(v, str):
            if v in b.EXACT:
                return b.EXACT[v]
            for old, new in b.STRINGS:
                v = v.replace(old, new)
            return v
        if isinstance(v, dict):
            return {k: m(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [m(x) for x in v]
        return v
    return m


def norm(r, m, drop_backend=False):
    d = {k: v for k, v in r.items() if k not in VOLATILE and not (drop_backend and k == 'backend')}
    if d.get('c_source') is not None:
        d['c_source'] = hashlib.sha256(d['c_source'].encode()).hexdigest()
    return json.loads(json.dumps(m(d)))          # as stored: tuples become lists


def programs(path, shard=None):
    rows = [json.loads(l) for l in open(path)]
    if shard:
        i, n = map(int, shard.split('/'))
        rows = rows[i::n]
    return rows


def score_rows(score_python_batch, rows, backend, max_steps, chunk=512):
    out = {}
    for truth in (False, True):
        part = [r for r in rows if r['truth'] == truth]
        for k in range(0, len(part), chunk):
            sub = part[k:k + chunk]
            res = score_python_batch([{'src': r['program'], 'inputs': {}, 'option_texts': r['options']} for r in sub],
                                     N=64, numbers='rational', max_steps=max_steps, boolean_options=truth,
                                     backend=backend)
            for r, o in zip(sub, res):
                out[r['key']] = o
    return out


def cmd_run(a):
    score_python_batch, _, _ = load_impl(a.impl, a.root)
    m = mapper() if a.impl != 'release' else (lambda v: v)
    rows = programs(a.programs, a.shard)
    res = score_rows(score_python_batch, rows, a.backend, a.max_steps)
    json.dump({k: norm(v, m, a.drop_backend) for k, v in res.items()}, open(a.out, 'w'), sort_keys=True)
    print(a.impl, a.backend, len(res), 'programs ->', a.out)


def cmd_compile(a):
    _, python_to_c, compile_c = load_impl(a.impl, a.root)
    out = {}
    for r in programs(a.programs, a.shard):
        try:
            c = python_to_c(r['program'])[0]
            cfg, mem, cmds, meta = compile_c(c, s=64, m=512, n=16384, N=64, checked=True, numbers='rational')
            h = hashlib.sha256(json.dumps([list(mem), [list(x) for x in cmds],
                                           sorted((k, list(v)) for k, v in meta['checks'].items())]).encode()).hexdigest()
            out[r['key']] = [hashlib.sha256(c.encode()).hexdigest(), h]
        except Exception as e:
            out[r['key']] = ['ERR', type(e).__name__]
    json.dump(out, open(a.out, 'w'), sort_keys=True)
    print(a.impl, 'compiled', len(out), 'errors', sum(v[0] == 'ERR' for v in out.values()))


def cmd_compare(a):
    A, B = ({} for _ in range(2))
    for paths, D in ((a.a, A), (a.b, B)):
        for p in paths.split(','):
            D.update(json.load(open(p)))
    keys = sorted(set(A) | set(B))
    diff = [k for k in keys if A.get(k) != B.get(k)]
    print(f'{len(keys)} programs, {len(keys) - len(diff)} identical, {len(diff)} differ')
    if a.out:
        json.dump([{'key': k, 'a': A.get(k), 'b': B.get(k)} for k in diff], open(a.out, 'w'), indent=1)
    for k in diff[:5]:
        print(k, '\n  a:', json.dumps(A.get(k))[:400], '\n  b:', json.dumps(B.get(k))[:400])
    sys.exit(1 if diff and a.strict else 0)


def cmd_iso(a):
    """In this process only the release is importable; expected results come from a file
    the repository wrote. n programs: singles, one batch, and permuted batches must all
    equal the expected results."""
    assert sys.flags.isolated, 'run with python -I'
    os.chdir(a.empty)
    assert not os.listdir('.'), 'cwd must be empty'
    sys.path.insert(0, a.root)
    import importlib.util
    for name in ('strandedcpu', 'loom_py', 'loom_api', 'c_compiler', 'loom_cuda', 'isa_interpreter', 'loom_v1'):
        assert importlib.util.find_spec(name) is None, f'{name} is importable'
    import xr
    from xr import score_python_batch
    rows = programs(a.programs)
    expected = json.load(open(a.expected))
    rows = random.Random(a.seed).sample([r for r in rows if r['key'] in expected], a.n)
    ident = lambda v: v
    log = {'python': sys.version.split()[0], 'version': xr.VERSION, 'backend': a.backend, 'n': len(rows)}
    single = {}
    for r in rows:
        single.update(score_rows(score_python_batch, [r], a.backend, a.max_steps))
    single = {k: norm(v, ident) for k, v in single.items()}
    log['single_equal_expected'] = sum(single[r['key']] == expected[r['key']] for r in rows)
    batch = {k: norm(v, ident) for k, v in score_rows(score_python_batch, rows, a.backend, a.max_steps).items()}
    log['batch_equal_single'] = sum(batch[r['key']] == single[r['key']] for r in rows)
    ok = 0
    for p in range(a.perms):
        perm = rows[:]
        random.Random(1000 + p).shuffle(perm)
        res = {k: norm(v, ident) for k, v in score_rows(score_python_batch, perm, a.backend, a.max_steps).items()}
        ok += all(res[r['key']] == single[r['key']] for r in rows)
    log['permutations_all_equal'] = f'{ok}/{a.perms}'
    loaded = sorted(n for n in sys.modules if n == 'xr' or n.startswith('xr.'))
    log['xr_modules'] = loaded
    assert all(sys.modules[n].__file__.startswith(a.root) for n in loaded)
    bad = sorted(n for n in sys.modules if any(s in n.lower() for s in ('loom', 'strand')))
    log['foreign_modules'] = bad
    if a.backend in ('cuda', 'auto'):
        import xr.gpu
        log['kernel_build'] = xr.gpu.BUILD.get('mode')
    print(json.dumps(log, indent=1))
    passed = (log['single_equal_expected'] == len(rows) and log['batch_equal_single'] == len(rows)
              and ok == a.perms and not bad)
    print('ISOLATION', 'PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'compile'):
        p = sub.add_parser(name)
        p.add_argument('--impl', required=True, choices=('package', 'flat', 'release'))
        p.add_argument('--root', required=True)
        p.add_argument('--programs', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--shard')
        p.add_argument('--backend', default='interpreter')
        p.add_argument('--max-steps', type=int, default=150_000)
        p.add_argument('--drop-backend', action='store_true')
    p = sub.add_parser('compare')
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--out')
    p.add_argument('--strict', action='store_true')
    p = sub.add_parser('iso')
    p.add_argument('--root', required=True)
    p.add_argument('--programs', required=True)
    p.add_argument('--expected', required=True)
    p.add_argument('--empty', required=True)
    p.add_argument('--backend', default='cuda')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--perms', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=150_000)
    a = ap.parse_args()
    {'run': cmd_run, 'compile': cmd_compile, 'compare': cmd_compare, 'iso': cmd_iso}[a.cmd](a)


if __name__ == '__main__':
    main()
