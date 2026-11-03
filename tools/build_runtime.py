"""Build the slim, name-neutral runtime release from a committed revision.

The release is the package xr: the Python front end (front), the C compiler
(cc), the exact ISA interpreter (isa), the batched CUDA executor (gpu), the
answer scorer (score) and the configuration and tables they share (ops, base,
ref). Sources are
read with git show, so a build depends only on the revision, the kernel image
and this script. Each module keeps only the top-level definitions reachable from
the scoring entry points; docstrings and comments are dropped; project names
are mapped to neutral ones, and the build fails if any banned name survives.

    python tools/build_runtime.py tree --rev REV --version V --out DIR [--fatbin F]
    python tools/build_runtime.py pack DIR OUT.tgz

tree without --fatbin writes the kernel-free tree; run `python -m xr.gpu
build-fatbin` in it on a machine with nvcc, then rebuild with --fatbin pointing
at the image (its .json must sit beside it). pack writes a byte-reproducible
tgz (sorted entries, zero mtimes and owners, gzip mtime 0).
"""
import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import symtable
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = 'xr'
LAYOUTS = {
    'package': {'front': 'strandedcpu/python.py', 'score': 'strandedcpu/api.py', 'cc': 'strandedcpu/compiler.py',
                'isa': 'strandedcpu/interpreter.py', 'gpu': 'strandedcpu/cuda.py', 'ops': 'strandedcpu/transformer.py',
                'ref': 'strandedcpu/reference.py', 'base': 'strandedcpu/subleq.py'},
    'flat': {'front': 'loom_py.py', 'score': 'loom_api.py', 'cc': 'c_compiler.py', 'isa': 'isa_interpreter.py',
             'gpu': 'loom_cuda.py', 'ops': 'loom_v1.py', 'ref': 'bayes_ref.py', 'base': 'subleq.py'},
}
MODULES = {'strandedcpu.python': 'front', 'strandedcpu.api': 'score', 'strandedcpu.compiler': 'cc',
           'strandedcpu.interpreter': 'isa', 'strandedcpu.cuda': 'gpu', 'strandedcpu.transformer': 'ops',
           'strandedcpu.reference': 'ref', 'strandedcpu.subleq': 'base', 'subleq': 'base', 'loom_py': 'front', 'loom_api': 'score', 'c_compiler': 'cc',
           'isa_interpreter': 'isa', 'loom_cuda': 'gpu', 'loom_v1': 'ops', 'extended_isa_v4': 'ops', 'bayes_ref': 'ref'}
# Derived from the upstream work (changed); the rest were written for the fork.
DERIVED = {'cc': 'c_compiler.py', 'ops': 'loom_v1.py', 'base': 'subleq.py', 'isa': 'demos/isa_interpreter.js'}
ROOTS = {'front': {'score_python_batch', 'score_python', 'python_to_c', 'PythonSubsetError'},
         'score': {'score_options_batch', 'score_options', 'parse_option_number', 'parse_option_truth', 'precompile'},
         'cc': {'compile_c', 'ExecutionFault'},
         'isa': {'ISAInterpreter'},
         'gpu': {'run_batch', 'available', 'Program', 'build_fatbin', 'source_sha256', 'BUILD', '_module'}}
# Backends this runtime leaves out: their bodies raise.
CUT = {'score': {'run_function', '_transformer', '_transformer_graph_run'}}
REVISION_FUNCS = {'_repository_revision', '_loom_commit'}
KEEP_MAIN = {'gpu'}
IDENTS = {'LoomFault': 'ExecutionFault', 'LoomConfig': 'Config', 'LoomComputer': 'Computer',
          'ExtendedConfigV4': 'Config', 'ExtendedNeuralComputerV4': 'Computer', 'init_state_v4': 'init_state',
          'read_memory_v4': 'read_memory', 'get_pc_v4': 'get_pc', 'loom_run': 'run',
          'LOOM_COMMIT': 'RUNTIME_VERSION', 'LOOM_DIFF_SHA256': 'RUNTIME_DIFF_SHA256',
          'STRANDEDCPU_COMMIT': 'RUNTIME_VERSION', 'STRANDEDCPU_DIFF_SHA256': 'RUNTIME_DIFF_SHA256'}
# Whole str / bytes constants replaced outright, then substrings, in order, in every
# str and bytes constant (docstrings are gone by then).
EXACT = {'loom.cu': 'kernel.cu', 'strandedcpu.cu': 'kernel.cu', 'loom_kernel.cu': 'kernel.cu',
         'strandedcpu_kernel.cu': 'kernel.cu', 'loom_kernel.fatbin': 'gpu.fatbin', 'kernel.fatbin': 'gpu.fatbin'}
STRINGS = [('python loom_cuda.py build-fatbin', 'python -m xr.gpu build-fatbin'),
           ('python -m strandedcpu.cuda build-fatbin', 'python -m xr.gpu build-fatbin'),
           ('loom_kernel', 'isa_kernel'), ('loom_run', 'run'),
           ('loom_cuda_ext4', 'xr_gpu_ext'), ('strandedcpu_cuda_ext', 'xr_gpu_ext'),
           ('LOOM_CUDA_BUILD', 'XR_GPU_BUILD'), ('STRANDEDCPU_CUDA_BUILD', 'XR_GPU_BUILD'),
           ('loom_commit', 'runtime_version'), ('strandedcpu_commit', 'runtime_version'),
           ('loom_diff_sha256', 'runtime_diff_sha256'), ('strandedcpu_diff_sha256', 'runtime_diff_sha256'),
           ('loom_cuda ', 'the GPU backend '), ('strandedcpu.cuda ', 'the GPU backend ')]
BANNED = re.compile(rb'loom|strand|turkcan|zxexz|jev|gemma|kgmem|esbar|marriott|anyscale|mkturk|liquamen|'
                    rb'plantwaterer|lulilla|eshbar', re.I)
HEADER = '# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE.\n'
CHANGED = '# Changed file: modified from the original work (see NOTICE).\n'


def git_show(rev, path):
    return subprocess.run(['git', '-C', REPO, 'show', f'{rev}:{path}'], check=True, capture_output=True).stdout


def layout_of(rev):
    files = subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', rev], check=True,
                           capture_output=True, text=True).stdout.split()
    return 'package' if 'strandedcpu/compiler.py' in files else 'flat'


# ---- transforms ---------------------------------------------------------------------------------

def strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]


def rename_idents(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in IDENTS:
            n.id = IDENTS[n.id]
        elif isinstance(n, ast.Attribute) and n.attr in IDENTS:
            n.attr = IDENTS[n.attr]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in IDENTS:
            n.name = IDENTS[n.name]
        elif isinstance(n, ast.alias):
            n.name = IDENTS.get(n.name, n.name)
            if n.asname:
                n.asname = IDENTS.get(n.asname, n.asname)
        elif isinstance(n, ast.arg) and n.arg in IDENTS:
            n.arg = IDENTS[n.arg]
        elif isinstance(n, ast.keyword) and n.arg in IDENTS:
            n.arg = IDENTS[n.arg]


def map_strings(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
            v = n.value
            text = v if isinstance(v, str) else v.decode('latin-1')
            if text in EXACT:
                n.value = EXACT[text] if isinstance(v, str) else EXACT[text].encode()
                continue
            for old, new in STRINGS:
                v = v.replace(old, new) if isinstance(v, str) else v.replace(old.encode(), new.encode())
            n.value = v


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


def module_aliases(tree):
    """Names bound to whole project modules anywhere in a module: name -> neutral module."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in MODULES:
                    out[a.asname or a.name] = MODULES[a.name] if a.asname or '.' not in a.name else None
    return {k: v for k, v in out.items() if v}


def is_main(stmt):
    return isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare) \
        and isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == '__name__'


def units_of(tree):
    """Top-level statements as (binds, stmt); imports split one alias per unit."""
    units = []
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                one = ast.ImportFrom(module=st.module, names=[a], level=st.level) if isinstance(st, ast.ImportFrom) \
                    else ast.Import(names=[a])
                bound = a.asname or (a.name if isinstance(st, ast.ImportFrom) else a.name.split('.')[0])
                units.append(({bound}, one))
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            units.append(({st.name}, st))
        else:
            bound = {t.id for t in ast.walk(st) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
            bound |= {d.name for d in ast.walk(st) if isinstance(d, (ast.FunctionDef, ast.ClassDef))}
            units.append((bound, st))
    return units


def refs_of(stmt, aliases):
    """(module or None, name) pairs a statement needs: None = same module."""
    need = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            need.add((None, n.id))
        elif isinstance(n, ast.ImportFrom) and n.module in MODULES:
            need |= {(MODULES[n.module], a.name) for a in n.names}
        elif isinstance(n, ast.Attribute):
            d = dotted(n.value)
            if d in MODULES:
                need.add((MODULES[d], n.attr))
            elif d in aliases:
                need.add((aliases[d], n.attr))
    return need


class ImportRewrite(ast.NodeTransformer):
    """Project imports -> the package; project-module attribute chains -> xr.<mod>."""

    def visit_ImportFrom(self, n):
        if n.module in MODULES:
            n.module = f'{PKG}.{MODULES[n.module]}'
        return n

    def visit_Import(self, n):
        for a in n.names:
            if a.name in MODULES:
                if not a.asname and '.' not in a.name:       # import loom_cuda -> import xr.gpu (uses rewritten)
                    a.name = f'{PKG}.{MODULES[a.name]}'
                else:
                    a.name = f'{PKG}.{MODULES[a.name]}'
        return n

    def visit_Attribute(self, n):
        d = dotted(n.value)
        if d in MODULES and '.' in d:                          # strandedcpu.cuda.X -> xr.gpu.X
            n.value = ast.Attribute(value=ast.Name(id=PKG, ctx=ast.Load()), attr=MODULES[d], ctx=ast.Load())
            return n
        self.generic_visit(n)
        return n

    def visit_Name(self, n):
        if n.id in MODULES and '.' not in n.id and n.id not in ('ref',):   # loom_cuda.X -> xr.gpu.X
            return ast.copy_location(ast.Attribute(value=ast.Name(id=PKG, ctx=ast.Load()), attr=MODULES[n.id],
                                                   ctx=n.ctx), n)
        return n


def cut_body(fn, why):
    fn.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='ValueError', ctx=ast.Load()),
                                      args=[ast.Constant(why)], keywords=[]), cause=None)]


def check_no_merge(before, after, mod):
    """Renaming must not merge two definitions into one name."""
    def defs(tree):
        out = []
        for scope in [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            names = [s.name for s in scope.body if isinstance(s, (ast.FunctionDef, ast.ClassDef))]
            out.append(sorted(names))
        return out
    for a, b in zip(defs(before), defs(after)):
        if len(set(a)) != len(set(b)):
            raise SystemExit(f'{mod}: renaming merged definitions: {sorted(set(a) - set(b))}')


def undefined_globals(src, name):
    """Global names a module reads that nothing in it binds (and are not builtins)."""
    import builtins
    top = symtable.symtable(src, name, 'exec')
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    used = set()

    def walk(t):
        for s in t.get_symbols():
            if (s.is_global() or t.get_type() == 'module') and s.is_referenced():
                used.add(s.get_name())
        for c in t.get_children():
            walk(c)
    walk(top)
    return sorted(u for u in used - bound if not hasattr(builtins, u) and u not in ('__name__', '__file__'))


# ---- build ------------------------------------------------------------------------------------------

def build_tree(rev, version, out, fatbin=None):
    layout = layout_of(rev)
    trees, aliases = {}, {}
    for mod, path in LAYOUTS[layout].items():
        t = ast.parse(git_show(rev, path))
        strip_docstrings(t)
        before = ast.parse(ast.unparse(t))
        rename_idents(t)
        check_no_merge(before, t, mod)
        for st in t.body:
            if isinstance(st, ast.FunctionDef) and st.name in CUT.get(mod, ()):
                cut_body(st, f'{st.name} is not included in this runtime')
            if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) \
                    and isinstance(st.value.func, ast.Name) and st.value.func.id in REVISION_FUNCS:
                st.value = ast.Tuple(elts=[ast.Constant(version), ast.Constant(None)], ctx=ast.Load())
        t.body = [st for st in t.body if not is_main(st) or mod in KEEP_MAIN]
        trees[mod] = t
        aliases[mod] = module_aliases(t)

    # closure over top-level units, starting from the entry points
    units = {mod: units_of(t) for mod, t in trees.items()}
    keep = {mod: set() for mod in trees}
    todo = [(mod, name) for mod, names in ROOTS.items() for name in names]
    todo += [(mod, None) for mod in KEEP_MAIN]
    seen = set()
    while todo:
        mod, name = todo.pop()
        if (mod, name) in seen:
            continue
        seen.add((mod, name))
        # A name nothing binds at top level is a local or a builtin; undefined_globals
        # below catches a real miss.
        hits = [k for k, (b, st) in enumerate(units[mod])
                if (name is None and is_main(st)) or (name is not None and name in b)]
        for k in hits:
            if k in keep[mod]:
                continue
            keep[mod].add(k)
            for m2, n2 in refs_of(units[mod][k][1], aliases[mod]):
                todo.append((m2 or mod, n2))
    # __future__ imports, and module-level expressions run for their effect, stay
    for mod in trees:
        for k, (b, st) in enumerate(units[mod]):
            if (isinstance(st, ast.ImportFrom) and st.module == '__future__') \
                    or (not b and not is_main(st) and isinstance(st, ast.Expr)):
                keep[mod].add(k)

    files = {}
    for mod in trees:
        body = []
        for k in sorted(keep[mod]):          # rejoin adjacent one-name imports from one module
            st = units[mod][k][1]
            if isinstance(st, ast.ImportFrom) and body and isinstance(body[-1], ast.ImportFrom) \
                    and body[-1].module == st.module and body[-1].level == st.level:
                body[-1] = ast.ImportFrom(module=st.module, names=body[-1].names + st.names, level=st.level)
            else:
                body.append(st)
        t = ast.Module(body=body, type_ignores=[])
        t = ImportRewrite().visit(t)
        map_strings(t)
        ast.fix_missing_locations(t)
        src = ast.unparse(t) + '\n'
        bad = undefined_globals(src, f'{mod}.py')
        if bad:
            raise SystemExit(f'{mod}: undefined globals after pruning: {bad}')
        files[f'{PKG}/{mod}.py'] = (HEADER + (CHANGED if mod in DERIVED else '') + '\n' + src).encode()
    files[f'{PKG}/__init__.py'] = (HEADER + '\n' + INIT.format(version=version)).encode()
    files['LICENSE'] = git_show(rev, 'LICENSE')
    files['NOTICE'] = NOTICE.encode()
    if fatbin:
        files[f'{PKG}/gpu.fatbin'] = open(fatbin, 'rb').read()
        files[f'{PKG}/gpu.fatbin.json'] = open(fatbin + '.json', 'rb').read()

    for path, data in files.items():
        for m in BANNED.finditer(data):
            if path in ('NOTICE',):
                continue
            raise SystemExit(f'{path}: banned name {m.group().decode(errors="replace")!r} at byte {m.start()}')
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f'{out} is not empty')
    for path, data in sorted(files.items()):
        p = os.path.join(out, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(data)
    manifest = {path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())}
    with open(os.path.join(out, 'SHA256SUMS'), 'w') as f:
        f.writelines(f'{h}  {p}\n' for p, h in manifest.items())
    kept = {mod: len(keep[mod]) for mod in trees}
    return {'rev': rev, 'layout': layout, 'version': version, 'units_kept': kept, 'files': manifest,
            'python': sys.version.split()[0]}


def pack(tree_dir, out_tgz):
    """A tgz whose bytes depend only on the tree's file contents."""
    paths = []
    for root, dirs, names in os.walk(tree_dir):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for nm in sorted(names):
            if not nm.endswith('.pyc'):
                paths.append(os.path.relpath(os.path.join(root, nm), tree_dir))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for d in sorted({os.path.dirname(p) for p in paths if os.path.dirname(p)}):
            ti = tarfile.TarInfo(d)
            ti.type, ti.mode, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = tarfile.DIRTYPE, 0o755, 0, 0, 0, '', ''
            tar.addfile(ti)
        for p in sorted(paths):
            data = open(os.path.join(tree_dir, p), 'rb').read()
            ti = tarfile.TarInfo(p)
            ti.size, ti.mode, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0o644, 0, 0, 0, '', ''
            tar.addfile(ti, io.BytesIO(data))
    with open(out_tgz, 'wb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0, compresslevel=9) as gz:
            gz.write(buf.getvalue())
    return hashlib.md5(open(out_tgz, 'rb').read()).hexdigest()


INIT = '''from xr.cc import ExecutionFault
from xr.front import score_python, score_python_batch
from xr.score import parse_option_number, parse_option_truth, score_options, score_options_batch
VERSION = {version!r}
__all__ = ['ExecutionFault', 'VERSION', 'parse_option_number', 'parse_option_truth', 'score_options',
           'score_options_batch', 'score_python', 'score_python_batch']
'''

NOTICE = '''This product includes software derived from LOOM
(https://github.com/mkturkcan/Loom) by Mehmet Kerem Turkcan and its
contributors, licensed under the Apache License, Version 2.0 (see LICENSE).
Upstream baseline: 71483281622684f32ea81c721b5a96c29cff4766
Paper: Loom: A Scalable Analytical Neural Computer Architecture,
https://arxiv.org/abs/2604.08816

Changed files derived from that work (each carries a notice that it was
changed):
  xr/cc.py   from c_compiler.py (the C compiler)
  xr/ops.py  from loom_v1.py (opcode numbers, word configuration and tables;
             the transformer itself is not included)
  xr/base.py from subleq.py (the base word and layout configuration)
  xr/isa.py  a Python port of demos/isa_interpreter.js (the ISA interpreter)

Changes: words of up to 64 bits; checked execution that stops on overflow,
division by zero and out-of-range access; exact rational numbers, roots and
powers; a Python-subset front end; a batched CUDA executor; answer-option
scoring; renamed modules; unused code and documentation strings removed.
The other files in xr/ are new and are licensed under the same terms.
'''


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tree')
    t.add_argument('--rev', required=True)
    t.add_argument('--version', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--fatbin')
    p = sub.add_parser('pack')
    p.add_argument('tree')
    p.add_argument('tgz')
    a = ap.parse_args()
    if a.cmd == 'tree':
        print(json.dumps(build_tree(a.rev, a.version, a.out, a.fatbin), indent=1))
    else:
        print(pack(a.tree, a.tgz))


if __name__ == '__main__':
    main()
