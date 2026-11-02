"""Compare JavaScript and Python ISA execution.

Checks memory, PC, carry, and steps for random instructions and compiled C.
The corpus includes both opcode pages, SUBLEQ, fixed point, and vector ops.
Requires Node.js.
"""

import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

import pytest

from tests import test_fuzz
from strandedcpu.compiler import compile_c
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.transformer import PAGE2_OPCODES

NODE = shutil.which('node')
JS = Path(__file__).resolve().parents[1] / 'javascript' / 'interpreter.js'
STEPS = 3000

REPLAY = """
const fs = require('fs');
const { ISAInterpreter } = require(process.argv[2]);
const out = JSON.parse(fs.readFileSync(process.argv[3])).map(t => {
    const vm = new ISAInterpreter(t.s, t.m, t.N, t.memory, t.commands, t.F);
    let steps = 0;
    while (steps < t.limit && !vm.step()) steps++;
    return { memory: vm.memory, pc: vm.pc, carry: vm.carry, steps };
});
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def python_run(s, m, N, memory, commands, F=2):
    vm = ISAInterpreter(s, m, N, list(memory), [tuple(c) for c in commands], F)
    steps = 0
    while steps < STEPS and not vm.step():
        steps += 1
    return {'memory': vm.memory, 'pc': vm.pc, 'carry': vm.carry, 'steps': steps}


def replay(cases):
    with tempfile.TemporaryDirectory() as d:
        src, inp, out = (os.path.join(d, f) for f in ('replay.js', 'in.json', 'out.json'))
        with open(src, 'w') as f:
            f.write(REPLAY)
        with open(inp, 'w') as f:
            json.dump([{**c, 'limit': STEPS} for c in cases], f)
        subprocess.run([NODE, src, JS, inp, out], check=True, timeout=600)
        with open(out) as f:
            return json.load(f)


def compare(cases):
    js = replay(cases)
    bad = [(k, c, j) for k, (c, j) in enumerate(zip(cases, js)) if j != c['want']]
    assert not bad, (f"{len(bad)} of {len(cases)} differ; first: N={bad[0][1]['N']} "
                     f"commands={bad[0][1]['commands'][:12]} js={ {k: v for k, v in bad[0][2].items() if k != 'memory'} } "
                     f"py={ {k: v for k, v in bad[0][1]['want'].items() if k != 'memory'} }")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_random_instructions_every_opcode():
    rng = random.Random(3)
    # not ARGMAX (2^mem[b] words), STORE (may write word 0), JMPI, VSETVL (added below, from word 0)
    ops = [o for o in range(1, 32) if o not in (20, 31)] + sorted(set(PAGE2_OPCODES) - {64, 67}) + [40, 41]
    cases = []
    for k in range(400):
        N = 8 if k % 2 else 16
        s, m, length = 32, 32, 40
        mem = [rng.randint(-(1 << (N - 1)), (1 << (N - 1)) - 1) if rng.random() < 0.5 else rng.randint(-9, 9)
               for _ in range(m)]
        mem[0] = rng.randint(0, 3)              # VSETVL reads word 0: vector length 2^0 .. 2^3;
        # every other operand is word 8 or above, outside any vector block that holds word 0
        cmds = []
        for i in range(length):
            op = rng.choice(ops)
            b, c = s + rng.randrange(8, m), s + rng.randrange(8, m)
            if op in (3, 4, 5, 10) or op >= 32 and op not in PAGE2_OPCODES:
                c = s + m + rng.randrange(i + 1, length + 1)        # forward jumps only
            if op >= 32 and op not in PAGE2_OPCODES:
                op = s + rng.randrange(m)                           # SUBLEQ: the subtrahend's address (32..63)
            if rng.random() < 0.1:
                cmds.append([67, s, 0])                             # VSETVL from word 0
            cmds.append([op, b, c])
        cmds.append([0, 0, 0])
        cases.append({'s': s, 'm': m, 'N': N, 'F': 2, 'memory': mem, 'commands': cmds,
                      'want': python_run(s, m, N, mem, cmds)})
    compare(cases)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_compiled_programs():
    rng = random.Random(9)
    cases = []
    for k in range(60):
        gen = test_fuzz.Gen(rng, 8, fix=k % 2 == 0)
        src = gen.program()
        try:
            cfg, mem, cmds, _ = compile_c(src, s=32, m=96, n=4096, N=8)
        except (SyntaxError, MemoryError, RuntimeError):
            continue
        cases.append({'s': cfg.s, 'm': cfg.m, 'N': cfg.N, 'F': cfg.log_frac_bits, 'memory': list(mem),
                      'commands': [list(c) for c in cmds], 'want': python_run(cfg.s, cfg.m, cfg.N, mem, cmds)})
    for k in range(20):                         # rational fix: noinline runtime calls through JMPI
        src = test_fuzz.FixGen(rng).program()
        cfg, mem, cmds, _ = compile_c(src, s=64, m=256, n=8192, N=16, numbers='rational')
        cases.append({'s': cfg.s, 'm': cfg.m, 'N': cfg.N, 'F': cfg.log_frac_bits, 'memory': list(mem),
                      'commands': [list(c) for c in cmds], 'want': python_run(cfg.s, cfg.m, cfg.N, mem, cmds)})
    assert len(cases) >= 60
    compare(cases)
