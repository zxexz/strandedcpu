"""Compare opcodes 21-31 with strandedcpu.reference on the transformer.

Checks the written words, untouched memory, carry, and PC. None of these
opcodes branches.
"""

import random
import torch

from strandedcpu.transformer import (
    Config, Computer, init_state, read_memory, get_pc, logadd_table,
    OP_HALT, OP_ADD, OP_SUB, OP_XOR, OP_MOV,
    OP_MAX, OP_LOGADD, OP_XLAT, OP_SHIFT, OP_ADC, OP_MULSTEP, OP_DIVSTEP,
    OP_LFSR, OP_CLZ, OP_ABS, OP_ARGMAX,
)
from strandedcpu.standard import StandardConfig, StandardComputer, init_state_standard
import strandedcpu.reference as ref

CFG8 = Config(s=32, m=32, n=128, N=8)   # m=32 leaves room for tables and blocks
CFG16 = Config(s=32, m=8, n=64, N=16)
SAMPLES = 40                                 # random cases per opcode at N=8

_computers = {}


def computer(cfg):
    key = (type(cfg).__name__, cfg.s, cfg.m, cfg.n, cfg.N, cfg.log_frac_bits)
    if key not in _computers:
        if isinstance(cfg, StandardConfig):
            _computers[key] = StandardComputer(cfg, use_argmax=True)
        else:
            _computers[key] = Computer(cfg)
    return _computers[key]


def run(cfg, memory, commands, carry=0, steps=None):
    """Run steps instructions; return (memory, carry, pc)."""
    comp = computer(cfg)
    if isinstance(cfg, StandardConfig):
        X = init_state_standard(cfg, memory, commands)
    else:
        X = init_state(cfg, memory, commands)
    X[cfg.idx_carry, 0] = float(carry)
    with torch.no_grad():
        for _ in range(steps or len(commands)):
            X = comp.step(X)
    flag = X[cfg.idx_carry, 0].item()
    assert flag in (0.0, 1.0), f"carry drifted to {flag}"
    return read_memory(X, cfg), int(flag), get_pc(X, cfg)


def rand_word(rng, N):
    return rng.randint(-(1 << (N - 1)), (1 << (N - 1)) - 1)


def edge_words(N):
    lo, hi = -(1 << (N - 1)), (1 << (N - 1)) - 1
    return [0, 1, -1, 2, -2, hi, lo, hi - 1, lo + 1, 5, -7]


def check_binary(cfg, opcode, expect, pairs, carry_in=None, expect_carry=None):
    """Run (opcode, s+0, s+1) on mem = [b, c, filler...] for each (b, c)."""
    s, N = cfg.s, cfg.N
    rng = random.Random(opcode)
    for b, c in pairs:
        filler = [rand_word(rng, N) for _ in range(cfg.m - 2)]
        cin = rng.randint(0, 1) if carry_in is None else carry_in
        mem, flag, pc = run(cfg, [b, c] + filler, [(opcode, s, s + 1), (OP_HALT, 0, 0)],
                            carry=cin, steps=1)
        want = [expect(b, c, cin)] + [c] + filler
        assert mem == want, f"op {opcode}: b={b} c={c} carry={cin}: got {mem[:2]}, want {want[:2]}"
        want_flag = cin if expect_carry is None else expect_carry(b, c, cin)
        assert flag == want_flag, f"op {opcode}: b={b} c={c}: carry {flag}, want {want_flag}"
        assert pc == s + cfg.m + 1, f"op {opcode}: b={b} c={c}: branched to {pc}"


def pairs_for(cfg, count=SAMPLES, seed=0):
    rng = random.Random(seed)
    edges = edge_words(cfg.N)
    out = [(b, c) for b in edges[:6] for c in edges[:6]]
    out += [(rng.choice(edges), rand_word(rng, cfg.N)) for _ in range(count // 4)]
    out += [(rand_word(rng, cfg.N), rand_word(rng, cfg.N)) for _ in range(count)]
    return out


# Single-instruction semantics

def test_max():
    check_binary(CFG8, OP_MAX, lambda b, c, _: ref.max_(b, c), pairs_for(CFG8))


def test_logadd():
    N, F = CFG8.N, CFG8.log_frac_bits
    table = logadd_table(F)
    pairs = pairs_for(CFG8)
    # every table entry and both window edges, in both orders
    for d in range(len(table) + 2):
        pairs += [(-40, -40 + d), (-40 + d, -40), (-100, -100 - d)]
    check_binary(CFG8, OP_LOGADD, lambda b, c, _: ref.logadd(b, c, N, F), pairs)


def test_shift():
    N = CFG8.N
    rng = random.Random(1)
    amounts = list(range(-N - 2, N + 3)) + [127, -128, 64, -64]
    pairs = [(rand_word(rng, N), a) for a in amounts for _ in range(3)]
    pairs += [(b, a) for b in (-1, -128, 127, 1) for a in amounts]
    check_binary(CFG8, OP_SHIFT, lambda b, c, _: ref.shift(b, c, N), pairs)


def test_clz():
    N = CFG8.N
    pairs = [(rand_word(random.Random(v), N), v) for v in range(-128, 128, 3)]
    pairs += [(0, 0), (5, 1 << (N - 2)), (5, -1)]
    check_binary(CFG8, OP_CLZ, lambda b, c, _: ref.clz(c, N), pairs)


def test_abs():
    N = CFG8.N
    pairs = [(rand_word(random.Random(v), N), v) for v in range(-128, 128, 3)] + [(0, -128), (0, 127)]
    check_binary(CFG8, OP_ABS, lambda b, c, _: ref.abs_(c, N), pairs)


def test_lfsr():
    N = CFG8.N
    check_binary(CFG8, OP_LFSR, lambda b, c, _: ref.lfsr(b, c, N), pairs_for(CFG8, seed=3))


def test_add_sub_set_carry():
    N = CFG8.N
    pairs = pairs_for(CFG8, count=20, seed=4)
    check_binary(CFG8, OP_ADD, lambda b, c, _: ref.wrap(b + c, N), pairs,
                 expect_carry=lambda b, c, _: ref.add_carry(b, c, N))
    check_binary(CFG8, OP_SUB, lambda b, c, _: ref.wrap(b - c, N), pairs,
                 expect_carry=lambda b, c, _: ref.sub_carry(b, c, N))


def test_adc():
    N = CFG8.N
    for cin in (0, 1):
        check_binary(CFG8, OP_ADC, lambda b, c, k: ref.adc(b, c, k, N)[0], pairs_for(CFG8, seed=5 + cin),
                     carry_in=cin, expect_carry=lambda b, c, k: ref.adc(b, c, k, N)[1])


def test_other_opcodes_keep_carry():
    s = CFG8.s
    for cin in (0, 1):
        mem, flag, _ = run(CFG8, [100, 100] + [0] * 30,
                           [(OP_XOR, s, s + 1), (OP_MOV, s + 2, s + 1), (OP_MAX, s, s + 1), (OP_HALT, 0, 0)],
                           carry=cin, steps=3)
        assert flag == cin


def test_xlat():
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(6)
    for table_size, base in ((8, 8), (16, 16), (4, 4), (8, 24)):
        table = [rand_word(rng, N) for _ in range(table_size)]
        for idx in list(range(table_size)) + [table_size + 1, -1]:
            mem = [rand_word(rng, N) for _ in range(cfg.m)]
            mem[base:base + table_size] = table
            mem[0] = idx
            col = ref.xlat_address(idx, s + base, s, N)
            want = list(mem)
            want[0] = mem[col - s]
            got, _, pc = run(cfg, mem, [(OP_XLAT, s, s + base), (OP_HALT, 0, 0)], steps=1)
            assert got == want, f"XLAT base={base} idx={idx}: got {got[0]}, want {want[0]}"
            assert pc == s + cfg.m + 1


def test_mulstep_single():
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(7)
    for _ in range(SAMPLES):
        hi, lo, mc = rand_word(rng, N), rand_word(rng, N), rand_word(rng, N)
        b = rng.choice([2, 3, 10, 11])          # pair (b, b^1), either orientation
        mem = [rand_word(rng, N) for _ in range(cfg.m)]
        mem[b], mem[b ^ 1], mem[20] = hi, lo, mc
        want = list(mem)
        want[b], want[b ^ 1] = ref.mulstep(hi, lo, mc, N)
        got, _, pc = run(cfg, mem, [(OP_MULSTEP, s + b, s + 20), (OP_HALT, 0, 0)], steps=1)
        assert got == want, f"MULSTEP hi={hi} lo={lo} mc={mc}: got {got[b], got[b^1]}, want {want[b], want[b^1]}"
        assert pc == s + cfg.m + 1


def test_divstep_single():
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(8)
    for _ in range(SAMPLES):
        d = rng.randint(1, (1 << N) - 1)
        rem = rng.randint(0, d - 1)              # invariant rem < divisor
        quo = rand_word(rng, N)
        b = rng.choice([4, 5, 12, 13])
        mem = [rand_word(rng, N) for _ in range(cfg.m)]
        mem[b], mem[b ^ 1], mem[21] = ref.wrap(rem, N), quo, ref.wrap(d, N)
        want = list(mem)
        want[b], want[b ^ 1] = ref.divstep(mem[b], quo, mem[21], N)
        got, _, pc = run(cfg, mem, [(OP_DIVSTEP, s + b, s + 21), (OP_HALT, 0, 0)], steps=1)
        assert got == want, f"DIVSTEP rem={rem} quo={quo} d={d}: got {got[b], got[b^1]}, want {want[b], want[b^1]}"
        assert pc == s + cfg.m + 1


def test_argmax():
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(9)
    cases = []
    for log_size, base in ((0, 8), (1, 8), (2, 12), (3, 8), (3, 16), (4, 16)):
        for trial in range(6):
            size = 1 << log_size
            if trial == 0:
                block = [7] * size                       # all tied: lowest index
            elif trial == 1:
                block = [-128] * size
                block[-1] = -127
            else:
                block = [rand_word(rng, N) for _ in range(size)]
                if trial == 2 and size > 2:
                    block[size - 1] = block[1] = max(block) + (1 if max(block) < 127 else 0)
            cases.append((log_size, base, block))
    for log_size, base, block in cases:
        mem = [rand_word(rng, N) for _ in range(cfg.m)]
        mem[base:base + len(block)] = block
        mem[0] = log_size
        want = list(mem)
        want[0] = ref.argmax(block)
        got, _, pc = run(cfg, mem, [(OP_ARGMAX, s, s + base), (OP_HALT, 0, 0)], steps=1)
        assert got == want, f"ARGMAX base={base} block={block}: got {got[0]}, want {want[0]}"
        assert pc == s + cfg.m + 1


# Multi-instruction sequences

def test_full_multiply():
    """N MULSTEPs give the full 2N-bit unsigned product."""
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(10)
    for a, b in [(255, 255), (0, 200), (1, 1), (128, 2)] + [(rng.randint(0, 255), rng.randint(0, 255)) for _ in range(6)]:
        mem = [0] * cfg.m
        mem[6], mem[7], mem[9] = 0, ref.wrap(b, N), ref.wrap(a, N)
        got, _, _ = run(cfg, mem, [(OP_MULSTEP, s + 6, s + 9)] * N, steps=N)
        product = (ref.unsigned(got[6], N) << N) | ref.unsigned(got[7], N)
        assert product == a * b, f"{a}*{b}: got {product}"


def test_full_divide():
    """N DIVSTEPs give quotient and remainder."""
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(11)
    for a, d in [(255, 1), (255, 255), (0, 7), (200, 13)] + [(rng.randint(0, 255), rng.randint(1, 255)) for _ in range(6)]:
        mem = [0] * cfg.m
        mem[14], mem[15], mem[9] = 0, ref.wrap(a, N), ref.wrap(d, N)
        got, _, _ = run(cfg, mem, [(OP_DIVSTEP, s + 14, s + 9)] * N, steps=N)
        assert (ref.unsigned(got[15], N), ref.unsigned(got[14], N)) == divmod(a, d), f"{a}/{d}: got q={got[15]} r={got[14]}"


def test_double_word_add_sub():
    """16-bit arithmetic from 8-bit words: ADD+ADC, and SUB + (XOR -1, ADC)."""
    cfg = CFG8
    s, N = cfg.s, cfg.N
    rng = random.Random(12)
    for _ in range(10):
        x, y = rng.randint(0, 0xFFFF), rng.randint(0, 0xFFFF)
        lo = lambda v: ref.wrap(v & 0xFF, N)
        hi = lambda v: ref.wrap(v >> 8, N)
        mem = [0] * cfg.m
        mem[0:5] = [lo(x), hi(x), lo(y), hi(y), -1]
        got, _, _ = run(cfg, mem, [(OP_ADD, s, s + 2), (OP_ADC, s + 1, s + 3)], steps=2)
        assert (ref.unsigned(got[1], N) << 8 | ref.unsigned(got[0], N)) == (x + y) & 0xFFFF
        program = [(OP_SUB, s, s + 2), (OP_XOR, s + 3, s + 4), (OP_ADC, s + 1, s + 3)]
        got, _, _ = run(cfg, mem, program, steps=3)
        assert (ref.unsigned(got[1], N) << 8 | ref.unsigned(got[0], N)) == (x - y) & 0xFFFF


def test_lfsr_sequence():
    """Galois LFSR with taps 0xB8 (x^8+x^6+x^5+x^4+1) over 40 steps."""
    cfg = CFG8
    s, N = cfg.s, cfg.N
    mem = [0] * cfg.m
    mem[0], mem[1] = 1, ref.wrap(0xB8, N)
    steps = 40
    got, _, _ = run(cfg, mem, [(OP_LFSR, s, s + 1)] * steps, steps=steps)
    state = 1
    for _ in range(steps):
        state = ref.lfsr(state, mem[1], N)
    assert got[0] == state


# Other configurations

def test_n16_spot_checks():
    """N=16: a handful of random cases per opcode."""
    cfg = CFG16
    s, N = cfg.s, cfg.N
    rng = random.Random(13)
    few = [(rand_word(rng, N), rand_word(rng, N)) for _ in range(6)] + [(-1, 1), (32767, -32768)]
    check_binary(cfg, OP_MAX, lambda b, c, _: ref.max_(b, c), few)
    check_binary(cfg, OP_LOGADD, lambda b, c, _: ref.logadd(b, c, N, cfg.log_frac_bits),
                 few + [(-300, -303), (-300, -297)])
    check_binary(cfg, OP_SHIFT, lambda b, c, _: ref.shift(b, c, N),
                 [(rand_word(rng, N), a) for a in (-17, -16, -15, -3, 0, 4, 15, 16, 40)])
    check_binary(cfg, OP_ADC, lambda b, c, k: ref.adc(b, c, k, N)[0], few, carry_in=1,
                 expect_carry=lambda b, c, k: ref.adc(b, c, k, N)[1])
    check_binary(cfg, OP_CLZ, lambda b, c, _: ref.clz(c, N), few + [(0, 0), (0, 1)])
    check_binary(cfg, OP_ABS, lambda b, c, _: ref.abs_(c, N), few)
    check_binary(cfg, OP_LFSR, lambda b, c, _: ref.lfsr(b, c, N), few)
    a, b = 51234, 40321
    mem = [0, b - 65536, 0, a - 65536, 0, 0, 0, 0]
    got, _, _ = run(cfg, mem, [(OP_MULSTEP, s, s + 3)] * N, steps=N)
    assert (ref.unsigned(got[0], N) << N | ref.unsigned(got[1], N)) == a * b
    mem = [0, a - 65536, 0, 1234, 0, 0, 0, 0]
    got, _, _ = run(cfg, mem, [(OP_DIVSTEP, s, s + 3)] * N, steps=N)
    assert (ref.unsigned(got[1], N), ref.unsigned(got[0], N)) == divmod(a, 1234)
    mem = [2, 0, 0, 0, 400, -9, 400, 3]
    got, _, _ = run(cfg, mem, [(OP_ARGMAX, s, s + 4), (OP_HALT, 0, 0)], steps=1)
    assert got[0] == 0


def test_standard_variant():
    """Broadcast-bias variant: its attention does not skip zero-key sources."""
    cfg = StandardConfig(s=32, m=32, n=128, N=8)
    s, N = cfg.s, cfg.N
    few = [(-5, 9), (100, -100), (-128, 127), (3, 3)]
    check_binary(cfg, OP_MAX, lambda b, c, _: ref.max_(b, c), few)
    check_binary(cfg, OP_LOGADD, lambda b, c, _: ref.logadd(b, c, N, cfg.log_frac_bits), few)
    check_binary(cfg, OP_ADC, lambda b, c, k: ref.adc(b, c, k, N)[0], few, carry_in=1,
                 expect_carry=lambda b, c, k: ref.adc(b, c, k, N)[1])
    mem = [3] + [0] * 31
    mem[8:16] = [4, -2, 9, 9, 0, 1, -50, 8]
    got, _, _ = run(cfg, mem, [(OP_ARGMAX, s, s + 8), (OP_HALT, 0, 0)], steps=1)
    assert got[0] == 2
    mem = [0] * 32
    mem[6], mem[7], mem[9] = 0, 77, 99
    got, _, _ = run(cfg, mem, [(OP_MULSTEP, s + 6, s + 9)] * N, steps=N)
    assert (ref.unsigned(got[6], N) << N | ref.unsigned(got[7], N)) == 77 * 99
    mem = [5] + [0] * 31
    mem[16:24] = list(range(10, 18))
    got, _, _ = run(cfg, mem, [(OP_XLAT, s, s + 16), (OP_HALT, 0, 0)], steps=1)
    assert got[0] == 15


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
    print(f"\nRESULTS: {passed}/{len(tests)} tests passed")


def test_argmax_n16_near_ties_exact():
    """ARGMAX's scores reach ~2^25 at N=16, past float32's exact integers;
    they are summed in float64, so neighbours of 32767 still separate."""
    cfg = CFG16
    s = cfg.s
    for block, want in (([32766, -32768, 32767, 32767], 2),
                        ([-32768, -32767, -32768, -32768], 1),
                        ([32765, 32766, -1, 0], 1),
                        ([-32768, -32768, -32768, -32768], 0)):
        mem = [2, 0, 0, 0] + block                   # VL word at 0, block at offsets 4..7
        got, _, _ = run(cfg, mem, [(OP_ARGMAX, s, s + 4), (OP_HALT, 0, 0)], steps=1)
        assert got[0] == want, (block, got[0])
