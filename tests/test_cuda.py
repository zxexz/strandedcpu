"""Compare CUDA execution and scoring with the interpreter.

Checks memory, steps, faults, and inexact-operation counts for raw instructions
and compiled C/Python at N=8..64. Binary fixed-point checks stop at N=32.
Requires a CUDA GPU.
"""

import random
from pathlib import Path

import pytest
import torch

from tests import test_fuzz
from tests import test_python
from strandedcpu.compiler import ExecutionFault, compile_c
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.api import score_options, score_options_batch
from strandedcpu.decision import score_row
from strandedcpu.python import python_to_c, score_python_batch
from strandedcpu.transformer import PAGE2_OPCODES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def py_run(s, m, N, memory, cmds, checks, max_steps):
    vm = ISAInterpreter(s, m, N, list(memory), [tuple(c) for c in cmds], 2, checks=checks)
    try:
        for steps in range(max_steps):
            if vm.step():
                return 'halted', steps + 1, vm.memory, None, vm.inexact
        return 'step_limit', max_steps, vm.memory, None, vm.inexact
    except ExecutionFault as f:
        return 'fault', vm.executed, vm.memory, (f.reason, f.detail), vm.inexact


def agree(s, m, N, progs, max_steps=20000):
    import strandedcpu.cuda
    got = strandedcpu.cuda.run_batch(s, m, N, progs, max_steps=max_steps)
    bad = []
    for k, (p, g) in enumerate(zip(progs, got)):
        status, steps, memory, fault, inexact = py_run(s, m, N, *p, max_steps)
        gf = None if g['fault'] is None else (g['fault'].reason, g['fault'].detail)
        if (status, steps, memory, fault, inexact) != (g['status'], g['steps'], g['memory'].tolist(), gf, g['inexact']):
            bad.append(k)
    assert not bad, f"{len(bad)} of {len(progs)} differ, first #{bad[0]}"


def test_random_instructions_every_opcode():
    rng = random.Random(1)
    ops = [o for o in range(1, 32) if o not in (20, 31)] + sorted(set(PAGE2_OPCODES) - {64, 67}) + [40, 41]
    for N in (8, 16, 32, 48, 63, 64):
        s, m, progs = 32, 32, []
        for _ in range(150):
            mem = [rng.randint(-(1 << (N - 1)), (1 << (N - 1)) - 1) if rng.random() < 0.5 else rng.randint(-9, 9)
                   for _ in range(m)]
            mem[0] = rng.randint(0, 3)          # VSETVL's word; other operands stay at word 8 and up
            cmds = []
            for i in range(40):
                op = rng.choice(ops)
                b, c = s + rng.randrange(8, m), s + rng.randrange(8, m)
                if op in (3, 4, 5, 10) or op >= 32 and op not in PAGE2_OPCODES:
                    c = s + m + rng.randrange(i + 1, 41)
                if op >= 32 and op not in PAGE2_OPCODES:
                    op = s + rng.randrange(m)
                if rng.random() < 0.1:
                    cmds.append([67, s, 0])
                cmds.append([op, b, c])
            cmds.append([0, 0, 0])
            progs.append((mem, cmds, None))
        agree(s, m, N, progs)


def test_compiled_programs_with_checks():
    rng = random.Random(2)
    for N in (8, 16, 64):
        progs = []
        for k in range(80):
            try:
                _, mem, cmds, meta = compile_c(test_fuzz.Gen(rng, N, fix=k % 2 == 0 and N <= 32).program(),
                                               s=64, m=128, n=4096, N=N, checked=True)
            except Exception:
                continue
            progs.append((list(mem), cmds, meta['checks']))
        agree(64, 128, N, progs)
    for N, numbers in ((16, 'binary'), (32, 'binary'), (32, 'rational'), (48, 'rational'), (64, 'rational')):
        progs = []
        for _ in range(60):
            _, mem, cmds, meta = compile_c(test_fuzz.FixGen(rng).program(), s=64, m=256, n=8192, N=N,
                                           checked=True, numbers=numbers)
            progs.append((list(mem), cmds, meta['checks']))
        agree(64, 256, N, progs, max_steps=200000)


def comparable(r):
    return {k: v for k, v in r.items() if k not in ('latency_s', 'backend')}


def test_scoring_apis_match_the_interpreter():
    rng = random.Random(3)
    items = []
    for k in range(120):
        src = test_python.Gen(rng, lists=k % 2 == 0).program()
        inputs = {v: rng.randint(-30, 30) for v in 'abc'}
        items.append({'src': src, 'inputs': inputs, 'option_texts': {'A': str(rng.randint(-50, 50)), 'B': '0'}})
    for N, numbers in ((32, 'binary'), (32, 'rational'), (64, 'rational')):
        cpu = score_python_batch(items, N=N, numbers=numbers)
        gpu = score_python_batch(items, N=N, numbers=numbers, backend='cuda')
        assert [comparable(r) for r in cpu] == [comparable(r) for r in gpu]
        assert {r['abstain_reason'] for r in cpu} > {None}          # a mix of outcomes was exercised
    prog = "fix price; fix r; int main() { r = price * 0.8; return 0; }"
    a = score_options(prog, {'price': 50}, {'A': '40'}, N=32, result='r', numbers='rational')
    b = score_options(prog, {'price': 50}, {'A': '40'}, N=32, result='r', numbers='rational', backend='cuda')
    assert comparable(a) == comparable(b) and b['backend'] == 'cuda' and b['status'] == 'ok'
    calls = [{'program': "int x; int main() { return x / 0; }", 'inputs': {'x': 1}, 'option_texts': {'A': '0'}},
             {'program': "int x; int main() { return x * x; }", 'inputs': {'x': 99999}, 'option_texts': {'A': '0'}},
             {'program': "int x; int main() { while (1) { x += 1; } return 0; }", 'option_texts': {'A': '0'},
              'max_steps': 500}]
    for c, g in zip(score_options_batch(calls, N=32), score_options_batch(calls, N=32, backend='cuda')):
        assert comparable(c) == comparable(g)
    row = [{'program': "def solution(price):\n    return price * 0.8\n", 'inputs': {'price': 50},
            'options': {'A': '40', 'B': '41'}},
           {'program': "def solution():\n    return not True or False\n", 'options': {'A': 'True', 'B': 'False'}},
           {'program': "int x; int main() { return x * 3; }", 'language': 'c', 'inputs': {'x': 7},
            'options': {'A': '21'}}]
    cpu, gpu = score_row(row), score_row(row, backend='cuda')
    assert [i['features'] for i in cpu['items']] == [i['features'] for i in gpu['items']]


def test_64_bit_exact_mode():
    """N=64 rational runs on the kernel and fits what N=32 overflows; binary
    fix above N=32 is refused by the kernel and runs on the interpreter."""
    import strandedcpu.cuda
    bayes = ("def solution():\n    pz = 0.37\n    n1 = pz * 0.62 * 0.81 + (1 - pz) * 0.21 * 0.44\n"
             "    d1 = pz * 0.62 + (1 - pz) * 0.21\n    n0 = pz * 0.38 * 0.52 + (1 - pz) * 0.79 * 0.29\n"
             "    d0 = pz * 0.38 + (1 - pz) * 0.79\n    return n1 / d1 - n0 / d0\n")
    item = [{'src': bayes, 'inputs': {}, 'option_texts': {'A': '771119393/2308731100'}}]
    assert score_python_batch(item, N=32, numbers='rational', backend='cuda')[0]['abstain_reason'] == 'overflow'
    r = score_python_batch(item, N=64, numbers='rational', backend='cuda')[0]
    assert r['status'] == 'ok' and r['backend'] == 'cuda' and 'fallback_reason' not in r
    prog = "fix price; fix r; int main() { r = price * 0.75; return 0; }"
    b = score_options(prog, {'price': 50}, {'A': '37.5'}, N=64, result='r', backend='cuda')
    assert b['status'] == 'ok' and b['backend'] == 'interpreter' and 'N=32' in b['fallback_reason']
    _, mem, cmds, meta = compile_c(prog, s=64, m=128, n=4096, N=64, checked=True)
    with pytest.raises(ValueError, match='N=32'):
        strandedcpu.cuda.run_batch(64, 128, 64, [(list(mem), cmds, meta['checks'])])


def test_transformer_audit_on_the_gpu():
    """backend='cuda-transformer': the kernel's result, checked against the
    looped transformer itself running on the GPU (N <= 16)."""
    prog = "fix x; fix r; fix h(fix v) { return v * v / 2; }\nint main() { r = h(x) - 0.25; return 0; }"
    r = score_options(prog, {'x': 1.5}, {'A': '0.875'}, N=16, result='r', numbers='rational',
                      backend='cuda-transformer', m=128)
    assert r['status'] == 'ok' and r['backend'] == 'cuda-transformer'
    row = score_row([{'program': "def solution(a):\n    return a * 3 + 1\n", 'inputs': {'a': 4},
                      'options': {'A': '13'}}], N=16, backend='cuda-transformer')
    assert row['items'][0]['result']['status'] == 'ok'
    with pytest.raises(ValueError):
        score_options("int x; int main() { return x; }", {'x': 1}, {'A': '1'}, N=32, backend='cuda-transformer')


@pytest.mark.parametrize('mode', ['prebuilt', 'nvrtc', 'nvrtc-ptx', 'nvcc', 'nvcc-ptx'])
def test_every_build_gives_the_same_results(mode):
    """Every way the kernel gets onto the GPU (the prebuilt fatbin; NVRTC or
    nvcc, for this GPU or as PTX the driver compiles at load time) must agree
    with the interpreter."""
    import os
    import subprocess
    import sys
    code = f"""
import random
from tests import test_fuzz
import strandedcpu.cuda
from strandedcpu.compiler import compile_c
from tests.test_cuda import agree
strandedcpu.cuda._module()
assert strandedcpu.cuda.BUILD['mode'] == '{mode}', strandedcpu.cuda.BUILD
rng = random.Random(5)
progs = []
for k in range(40):
    try:
        _, mem, cmds, meta = compile_c(test_fuzz.Gen(rng, 16, fix=k % 2 == 0).program(), s=64, m=128, n=4096,
                                       N=16, checked=True)
    except Exception:           # a random program that does not compile
        continue
    progs.append((list(mem), cmds, meta['checks']))
exact = []
for _ in range(20):
    _, mem, cmds, meta = compile_c(test_fuzz.FixGen(rng).program(), s=64, m=256, n=8192, N=32, checked=True,
                                   numbers='rational')
    exact.append((list(mem), cmds, meta['checks']))
agree(64, 128, 16, progs)
agree(64, 256, 32, exact, max_steps=200000)
print('build ok')
"""
    here = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, '-c', code], cwd=here, env={**os.environ, 'STRANDEDCPU_CUDA_BUILD': mode},
                         capture_output=True, text=True, timeout=900)
    assert 'build ok' in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


def test_roots_on_the_gpu():
    """The kernel runs the root and power runtimes and stops where the
    interpreter does: a stop is a division by a zero word whose check names
    the reason."""
    cases = test_python.ROOT_CASES + test_python.POW_CASES
    rows = [{'src': src, 'inputs': {}, 'option_texts': {'A': opt}} for src, opt, _, _ in cases]
    cpu = score_python_batch(rows, N=64, numbers='rational', backend='interpreter')
    gpu = score_python_batch(rows, N=64, numbers='rational', backend='cuda')
    for (src, _, status, reason), c, g in zip(cases, cpu, gpu):
        assert g['backend'] == 'cuda', src
        assert (g['status'], g['abstain_reason']) == (c['status'], c['abstain_reason']) == (status, reason), src
        if status == 'ok':
            assert (g['value'], g['steps']) == (c['value'], c['steps']), src
