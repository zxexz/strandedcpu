"""Compare page-2 transformer instructions with the interpreter.

Checks memory, per-column carries, and PC, plus reduction, Walsh-Hadamard,
and CALL/RET programs. An exhaustive operand-pair test covers L4-L7 at a
memory column.
"""

import random
import torch

from strandedcpu.transformer import (
    Config, Computer, init_state, read_memory, get_pc, opcode_pattern, logadd_table,
    OP_HALT, OP_MOV, OP_INC, OP_JMP, OP_JMPI, OP_FALIGN, OP_FNORM, OP_VSETVL,
    OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD, OP_VCADD, OP_VFILL, OP_VSHIFT,
    OP_VXLAT, OP_VREV, OP_VTRANS, OP_VRAND, OP_VPERMX, OP_VADDX, OP_VBFLY, OP_VMAXX,
    OP_VLOGADDX, VECTOR_OPCODES,
)
from strandedcpu.standard import StandardConfig, StandardComputer, init_state_standard
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.subleq import to_bipolar
import strandedcpu.reference as ref

CFG = Config(s=32, m=32, n=128, N=8)
BLOCK_SOURCE = (OP_VMOV, OP_VADD, OP_VSUB, OP_VMAX, OP_VLOGADD, OP_VCADD, OP_VREV, OP_VTRANS)
PARTNER = (OP_VPERMX, OP_VADDX, OP_VMAXX, OP_VLOGADDX)
_computers = {}


def machine(cfg):
    key = (type(cfg).__name__, cfg.s, cfg.m, cfg.n, cfg.N)
    if key not in _computers:
        std = isinstance(cfg, StandardConfig)
        _computers[key] = StandardComputer(cfg, use_argmax=True) if std else Computer(cfg)
    return _computers[key]


def compare(cfg, memory, commands, steps, vcarry=None):
    """Run steps instructions on both; assert identical state; return the interpreter."""
    std = isinstance(cfg, StandardConfig)
    X = (init_state_standard if std else init_state)(cfg, memory, commands)
    vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, memory, commands, cfg.log_frac_bits)
    if vcarry is not None:
        X[cfg.idx_vcarry, cfg.s:cfg.s + cfg.m] = torch.tensor(vcarry, dtype=torch.float32)
        vm.vcarry = list(vcarry)
    comp = machine(cfg)
    with torch.no_grad():
        for _ in range(steps):
            X = comp.step(X)
            vm.step()
    assert read_memory(X, cfg) == vm.memory, f"memory {read_memory(X, cfg)} != {vm.memory}"
    outside = torch.cat([X[cfg.idx_memory:cfg.idx_memory + cfg.N, :cfg.s],
                         X[cfg.idx_memory:cfg.idx_memory + cfg.N, cfg.s + cfg.m:]], dim=1)
    assert torch.all(outside == 0), "memory rows written outside the memory columns"
    carries = X[cfg.idx_vcarry, cfg.s:cfg.s + cfg.m].tolist()
    assert carries == [float(c) for c in vm.vcarry], "per-column carries differ"
    assert get_pc(X, cfg) == vm.pc
    return vm


def vector_case(rng, cfg, opcode, t):
    """Random memory and one VSETVL + opcode program for vector length 2^t."""
    s, N = cfg.s, cfg.N
    vl = 1 << t
    mem = [rng.randint(-128, 127) for _ in range(cfg.m)]
    b = rng.choice([8, 16])
    mem[2] = t
    c = 3
    if opcode in BLOCK_SOURCE:
        c = rng.choice([0, 16, 24, b]) if vl <= 8 else rng.choice([0, 16])
    elif opcode in PARTNER:
        mem[3] = rng.randint(0, 2 * vl)
    elif opcode == OP_VBFLY:
        mem[3] = 1 << rng.randint(0, max(t - 1, 0))
    elif opcode == OP_VSHIFT:
        mem[3] = rng.choice([rng.randint(-9, 9), 127, -128])
    elif opcode == OP_VXLAT:
        mem[b:b + vl] = [rng.randint(-3, 9) for _ in range(vl)]
        c = 24
    return mem, [(OP_VSETVL, s + 2, 0), (opcode, s + b, s + c), (OP_HALT, 0, 0)]


# Single instructions

def test_every_vector_opcode():
    rng = random.Random(0)
    for opcode in VECTOR_OPCODES:
        for t in (0, 1, 2, 3, 4):
            for _ in range(2):
                mem, cmds = vector_case(rng, CFG, opcode, t)
                compare(CFG, mem, cmds, 2, vcarry=[rng.randint(0, 1) for _ in range(CFG.m)])


def test_float_assists_and_jmpi():
    rng = random.Random(1)
    s = CFG.s
    for _ in range(40):
        mem = [rng.randint(-128, 127) for _ in range(CFG.m)]
        mem[5] = rng.randint(0, 40)
        mem[9] = mem[6] + rng.randint(-10, 10) if rng.random() < 0.7 else mem[9]
        mem[9] = max(-128, min(127, mem[9]))
        for program in ([(OP_FALIGN, s + 6, s + 9), (OP_HALT, 0, 0)],
                        [(OP_FNORM, s + 7, s + 9), (OP_HALT, 0, 0)],
                        [(OP_JMPI, 0, s + 5), (OP_HALT, 0, 0)]):
            compare(CFG, mem, program, 1)
    for m_val in (0, -1, 1, 63, 64, -64, -65, -128, 127, 5, -5):
        mem = [0] * CFG.m
        mem[6], mem[7] = 3, m_val
        compare(CFG, mem, [(OP_FNORM, s + 6, 0), (OP_HALT, 0, 0)], 1)


def test_call_return_through_jmpi():
    """A subroutine (INC mem[0]) called from two sites; mem[1] holds the
    return instruction index."""
    s, m = CFG.s, CFG.m
    sub = 7
    program = [
        (OP_MOV, s + 1, s + 3),   # 0: ret = 2
        (OP_JMP, 0, s + m + sub), # 1: call
        (OP_MOV, s + 1, s + 4),   # 2: ret = 4
        (OP_JMP, 0, s + m + sub), # 3: call
        (OP_INC, s + 2, 0),       # 4: mark
        (OP_HALT, 0, 0),          # 5
        (OP_HALT, 0, 0),          # 6
        (OP_INC, s + 0, 0),       # 7: subroutine body
        (OP_JMPI, 0, s + 1),      # 8: return
    ]
    mem = [10, 0, 0, 2, 4] + [0] * (m - 5)
    vm = compare(CFG, mem, program, 10)
    assert vm.memory[0] == 12 and vm.memory[2] == 1 and vm.pc == 0


def test_standard_variant():
    cfg = StandardConfig(s=32, m=32, n=128, N=8)
    rng = random.Random(2)
    for opcode in VECTOR_OPCODES:
        mem, cmds = vector_case(rng, cfg, opcode, rng.choice([1, 2, 3]))
        compare(cfg, mem, cmds, 2)


# Reductions and transforms

def test_hypercube_all_reduce():
    """After log2(16) stages every column holds the reduction of the block."""
    s = CFG.s
    rng = random.Random(3)
    vals = [rng.randint(-128, 127) for _ in range(16)]
    stages = [1, 2, 4, 8]
    mem = [0] * CFG.m
    mem[0] = 4
    mem[1:5] = stages
    mem[16:32] = vals

    # MAX and LOGADD
    for opcode, combine in ((OP_VMAXX, max), (OP_VLOGADDX, lambda a, b: ref.logadd(a, b, 8, 2))):
        program = [(OP_VSETVL, s + 0, 0)] + [(opcode, s + 16, s + 1 + k) for k in range(4)] + [(OP_HALT, 0, 0)]
        vm = compare(CFG, mem, program, 5)
        expect = list(vals)
        for m_ in stages:
            expect = [combine(expect[i], expect[i ^ m_]) for i in range(16)]
        assert vm.memory[16:32] == expect
        if opcode == OP_VMAXX:
            assert expect == [max(vals)] * 16

    # Widen the sum to two words: VL=8, low words at 16..23, high words at 24..31.
    lo = [rng.randint(-128, 127) for _ in range(8)]
    mem = [0] * CFG.m
    mem[0], mem[1], mem[2], mem[3], mem[4] = 3, 1, 2, 4, -7
    mem[16:24] = lo
    program = [(OP_VSETVL, s + 0, 0),
               (OP_VMOV, s + 24, s + 16), (OP_VSHIFT, s + 24, s + 4)]  # hi = sign of lo
    for k in (1, 2, 3):
        program += [(OP_VADDX, s + 16, s + k),   # lo += lo partner (carry per column)
                    (OP_VADDX, s + 24, s + k),   # hi += hi partner
                    (OP_VCADD, s + 24, s + 16)]  # hi += the pair's carry, once
    program.append((OP_HALT, 0, 0))
    vm = compare(CFG, mem, program, len(program) - 1)
    total = sum(lo)
    for i in range(8):
        assert (ref.unsigned(vm.memory[24 + i], 8) << 8 | ref.unsigned(vm.memory[16 + i], 8)) == total & 0xFFFF


def test_walsh_hadamard():
    s = CFG.s
    vals = [3, -1, 4, 1, -5, 9, -2, 6]
    mem = [0] * CFG.m
    mem[0], mem[1], mem[2], mem[3] = 3, 1, 2, 4
    mem[8:16] = vals
    program = [(OP_VSETVL, s + 0, 0)] + [(OP_VBFLY, s + 8, s + k) for k in (1, 2, 3)] + [(OP_HALT, 0, 0)]
    vm = compare(CFG, mem, program, 4)
    H = [[(-1) ** bin(i & j).count('1') for j in range(8)] for i in range(8)]
    assert vm.memory[8:16] == [ref.wrap(sum(H[i][j] * vals[j] for j in range(8)), 8) for i in range(8)]


def test_vrand_statistics():
    """VRAND words over many (seed, column) pairs: every value appears, the
    low bit is balanced, and neighbouring columns are not correlated."""
    N = 8
    words = [ref.vrand(seed, col, N) for seed in range(256) for col in range(32)]
    counts = [0] * 256
    for w in words:
        counts[ref.unsigned(w, N)] += 1
    assert min(counts) > 0
    expected = len(words) / 256
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    assert chi2 < 400                     # 255 degrees of freedom
    lows = [ref.unsigned(w, N) & 1 for w in words]
    assert abs(sum(lows) / len(lows) - 0.5) < 0.02
    a = [ref.unsigned(ref.vrand(seed, col, N), N) for seed in range(256) for col in range(31)]
    b = [ref.unsigned(ref.vrand(seed, col + 1, N), N) for seed in range(256) for col in range(31)]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    var = sum((x - ma) ** 2 for x in a) / len(a)
    assert abs(cov / var) < 0.05


# Exhaustive per-column datapath (L4-L7 FFNs at a memory column)

def test_datapath_exhaustive():
    cfg = CFG
    N, logn, s = cfg.N, cfg.logn, cfg.s
    layers = Computer(cfg).layers
    col = s + 8                      # the column under test, in block b = s + 8, VL = 8
    vals = torch.arange(-128, 128)
    x, y = (t.flatten() for t in torch.meshgrid(vals, vals, indexing="ij"))
    K = x.numel()

    def bip(v, bits):
        u = v.to(torch.int64) & ((1 << bits) - 1)
        sh = torch.arange(bits - 1, -1, -1)
        return ((u.unsqueeze(0) >> sh.unsqueeze(1)) & 1).float() * 2 - 1

    def run(opcode, scalar=None, gather=True, ycarry=None):
        X = torch.zeros(cfg.d_model, K)
        X[cfg.idx_memory:cfg.idx_memory + N] = bip(x, N)
        X[cfg.idx_pos_enc:cfg.idx_pos_enc + logn] = to_bipolar(col, logn).unsqueeze(1)
        X[cfg.idx_tag:cfg.idx_tag + N] = bip(torch.full((K,), col - s), N)
        X[cfg.idx_memcol] = 1.0
        X[0:logn] = opcode_pattern(opcode, logn).unsqueeze(1)                  # payload
        X[logn:2 * logn] = to_bipolar(s + 8, logn).unsqueeze(1)
        X[2 * logn:3 * logn] = to_bipolar(s + 16, logn).unsqueeze(1)
        lt = cfg.idx_buffer + 4 * N
        X[lt + logn - 3:lt + logn] = 1.0                                     # VL = 8
        if scalar is not None:
            X[cfg.idx_buffer + N:cfg.idx_buffer + 2 * N] = 0.5 * bip(scalar, N)
        ffn = lambda layer, X: X + layer.W2 @ torch.relu(layer.W1 @ X + layer.b1[:, col:col + 1]) + layer.b2[:, col:col + 1]
        X = ffn(layers[3], X)                                                # L4
        if gather:                                                           # L5 attention
            X[cfg.idx_scratchpad:cfg.idx_scratchpad + N] = bip(y, N)
        if ycarry is not None:
            X[lt + 3] = ycarry
        X[cfg.idx_buffer:cfg.idx_buffer + N] = X[cfg.idx_memory:cfg.idx_memory + N]   # bystander reads
        X[cfg.idx_buffer + 3 * N:cfg.idx_buffer + 4 * N] = X[cfg.idx_memory:cfg.idx_memory + N]
        for layer in layers[4:7]:                                            # L5, L6, L7
            X = ffn(layer, X)
        out = X[cfg.idx_memory:cfg.idx_memory + N]
        assert torch.all(out.abs() == 1.0), "memory bits are not exactly +/-1"
        bits = (out > 0).to(torch.int64)
        u = (bits * (2 ** torch.arange(N - 1, -1, -1)).unsqueeze(1)).sum(0)
        assert torch.all(X[0:3 * logn] == 0), "payload not cleared"
        return torch.where(u >= 128, u - 256, u), X[cfg.idx_vcarry]

    w = lambda v: ((v + 128) % 256) - 128
    got, carry = run(OP_VADD)
    assert torch.equal(got, w(x + y)) and torch.equal(carry, (((x & 255) + (y & 255)) >> 8).float())
    got, carry = run(OP_VSUB)
    assert torch.equal(got, w(x - y)) and torch.equal(carry, ((x & 255) >= (y & 255)).float())
    got, _ = run(OP_VMAX)
    assert torch.equal(got, torch.maximum(x, y))
    table = logadd_table(cfg.log_frac_bits)
    d = (x - y).abs()
    corr = torch.tensor([table[i] if i < len(table) else 0 for i in range(256)])[d]
    got, _ = run(OP_VLOGADD)
    assert torch.equal(got, w(torch.maximum(x, y) + corr))
    got, _ = run(OP_VADDX)
    assert torch.equal(got, w(x + y))
    for ycarry in (0.0, 1.0):
        got, carry = run(OP_VCADD, gather=False, ycarry=ycarry)
        assert torch.equal(got, w(x + int(ycarry)))
    for m_bit, upper in ((1, False), (8, True)):        # column 40 = 0b101000: bit 3 set
        got, _ = run(OP_VBFLY, scalar=torch.full((K,), m_bit))
        assert torch.equal(got, w(y - x) if upper else w(x + y)), m_bit
    got, _ = run(OP_VSHIFT, scalar=y, gather=False)
    assert torch.equal(got, torch.tensor([ref.shift(a, b, 8) for a, b in zip(x.tolist(), y.tolist())]))
    got, _ = run(OP_VFILL, scalar=y, gather=False)
    assert torch.equal(got, y)
    got, _ = run(OP_VRAND, scalar=y, gather=False)
    assert torch.equal(got, torch.tensor([ref.vrand(b, col - s, 8) for b in y.tolist()]))
