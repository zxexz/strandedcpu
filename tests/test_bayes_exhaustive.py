"""Check scalar arithmetic FFNs against strandedcpu.reference.

Each column holds an independent post-attention operand case for L2-L4.
N=8 covers all operand pairs; N=16 samples larger threshold weights.
Tests in test_bayes.py cover the attention paths and memory writes.
"""

import random
import torch

from strandedcpu.transformer import (
    Config, Computer, opcode_pattern,
    OP_ADD, OP_SUB, OP_MAX, OP_LOGADD, OP_SHIFT, OP_ADC, OP_MULSTEP, OP_DIVSTEP,
    OP_LFSR, OP_CLZ, OP_ABS, OP_FALIGN, OP_FNORM,
)
from strandedcpu.subleq import to_bipolar
import strandedcpu.reference as ref


def _bipolar(values, N):
    """values (K,) signed ints -> (N, K) bipolar, MSB first."""
    u = torch.as_tensor(values, dtype=torch.int64) & ((1 << N) - 1)
    shifts = torch.arange(N - 1, -1, -1, dtype=torch.int64)
    return ((u.unsqueeze(0) >> shifts.unsqueeze(1)) & 1).float() * 2 - 1


def _decode(rows, N):
    """(N, K) +/-1 -> signed ints; fails on anything that is not exactly +/-1."""
    assert torch.all(rows.abs() == 1.0), "result bits are not exactly +/-1"
    bits = (rows > 0).to(torch.int64)
    weights = 2 ** torch.arange(N - 1, -1, -1, dtype=torch.int64)
    u = (bits * weights.unsqueeze(1)).sum(0)
    return torch.where(u >= 1 << (N - 1), u - (1 << N), u).tolist()


class FFNHarness:
    """Column-0 FFNs of L2, L3, L4 applied to K independent cases."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.layers = Computer(cfg).layers[1:4]

    def run(self, opcode, b_mem, c_mem, partner=None, carry=None, b_col=None, c_col=None):
        cfg = self.cfg
        N, logn = cfg.N, cfg.logn
        K = len(b_mem)
        b_col = cfg.s + 2 if b_col is None else b_col
        c_col = cfg.s + 9 if c_col is None else c_col
        X = torch.zeros(cfg.d_model, K)
        X[-1] = 1.0
        cmd = cfg.idx_scratch_cmd
        X[cmd:cmd + logn] = opcode_pattern(opcode, logn).unsqueeze(1)
        X[cmd + logn:cmd + 2 * logn] = to_bipolar(b_col, logn).unsqueeze(1)
        X[cmd + 2 * logn:cmd + 3 * logn] = to_bipolar(c_col, logn).unsqueeze(1)
        buf = cfg.idx_buffer
        if partner is not None:
            X[buf:buf + N] = 0.5 * _bipolar(partner, N)
            # what L1 parks in load_temp for the pair read; L2 must clear it
            aux = to_bipolar(b_col ^ 1, logn) - opcode_pattern(opcode, logn)
            X[buf + 4 * N:buf + 4 * N + logn] = aux.unsqueeze(1)
        X[buf + N:buf + 2 * N] = 0.5 * _bipolar(b_mem, N)
        X[buf + 2 * N:buf + 3 * N] = 0.5 * _bipolar(c_mem, N)
        if carry is not None:
            X[cfg.idx_carry] = torch.as_tensor(carry, dtype=torch.float32)
        scr_sub = cfg.idx_scratchpad
        scr_min = cfg.idx_scratchpad + N

        def ffn(layer, X):
            with torch.no_grad():
                return X + layer.W2 @ torch.relu(layer.W1 @ X + layer.b1[:, :1]) + layer.b2[:, :1]

        # L2 must leave operand bits at +/-1, buffer bits at +/-0.5,
        # and carry at 0/1 before L3/L4.
        X = ffn(self.layers[0], X)
        _decode(X[scr_sub:scr_sub + N], N)
        _decode(X[scr_min:scr_min + N], N)
        second = _decode(2 * X[buf + 2 * N:buf + 3 * N], N)   # L5 head 2 writes 2*buf_c
        flag = X[cfg.idx_carry]
        assert torch.all((flag == 0) | (flag == 1)), "carry is not exactly 0/1"
        if partner is not None:
            _decode(X[cmd + 2 * logn:cmd + 3 * logn], logn)   # addr_c retargeted
            assert torch.all(X[cmd + 2 * logn:cmd + 3 * logn] == to_bipolar(b_col ^ 1, logn).unsqueeze(1))
        assert torch.all(X[buf + 4 * N:buf + 4 * N + logn] == 0), "load_temp not cleared"
        # The snap and subtract stages must preserve bipolar output bits.
        X = ffn(self.layers[2], ffn(self.layers[1], X))
        result = _decode(X[scr_min:scr_min + N], N)
        return result, second, flag.to(torch.int64).tolist()


def _all_pairs(N):
    v = torch.arange(-(1 << (N - 1)), 1 << (N - 1), dtype=torch.int64)
    b, c = torch.meshgrid(v, v, indexing="ij")
    return b.flatten().tolist(), c.flatten().tolist()


def _check(name, got, want, b, c):
    bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
    assert not bad, f"{name}: {len(bad)} mismatches, first b={b[bad[0]]} c={c[bad[0]]}: got {got[bad[0]]}, want {want[bad[0]]}"


def check_binary_ops(cfg, b, c, carry):
    N, F = cfg.N, cfg.log_frac_bits
    h = FFNHarness(cfg)
    cases = [
        (OP_MAX, lambda x, y, k: ref.max_(x, y), None),
        (OP_LOGADD, lambda x, y, k: ref.logadd(x, y, N, F), None),
        (OP_SHIFT, lambda x, y, k: ref.shift(x, y, N), None),
        (OP_LFSR, lambda x, y, k: ref.lfsr(x, y, N), None),
        (OP_CLZ, lambda x, y, k: ref.clz(y, N), None),
        (OP_ABS, lambda x, y, k: ref.abs_(y, N), None),
        (OP_ADD, lambda x, y, k: ref.wrap(x + y, N), lambda x, y, k: ref.add_carry(x, y, N)),
        (OP_SUB, lambda x, y, k: ref.wrap(x - y, N), lambda x, y, k: ref.sub_carry(x, y, N)),
        (OP_ADC, lambda x, y, k: ref.adc(x, y, k, N)[0], lambda x, y, k: ref.adc(x, y, k, N)[1]),
    ]
    for opcode, value, flag in cases:
        result, _, got_flag = h.run(opcode, b, c, carry=carry)
        _check(f"op {opcode}", result, [value(x, y, k) for x, y, k in zip(b, c, carry)], b, c)
        want_flag = carry if flag is None else [flag(x, y, k) for x, y, k in zip(b, c, carry)]
        _check(f"op {opcode} carry", got_flag, want_flag, b, c)


def check_pair_ops(cfg, x, y, partner):
    N = cfg.N
    h = FFNHarness(cfg)
    hi_lo = [ref.mulstep(a, p, m, N) for a, p, m in zip(x, partner, y)]
    result, second, _ = h.run(OP_MULSTEP, x, y, partner=partner)
    _check("MULSTEP hi", result, [t[0] for t in hi_lo], x, y)
    _check("MULSTEP lo", second, [t[1] for t in hi_lo], x, y)
    rem_quo = [ref.divstep(r, q, d, N) for r, q, d in zip(x, partner, y)]
    result, second, _ = h.run(OP_DIVSTEP, x, y, partner=partner)
    _check("DIVSTEP rem", result, [t[0] for t in rem_quo], x, y)
    _check("DIVSTEP quo", second, [t[1] for t in rem_quo], x, y)


def test_n8_binary_exhaustive():
    cfg = Config(s=32, m=32, n=128, N=8)
    b, c = _all_pairs(8)
    for carry_in in (0, 1):
        check_binary_ops(cfg, b, c, [carry_in] * len(b))


def test_n8_pair_exhaustive():
    """All (mem[b], mem[c]) for MULSTEP/DIVSTEP, with partner words that cover
    every bit the step reads from mem[b^1]."""
    cfg = Config(s=32, m=32, n=128, N=8)
    b, c = _all_pairs(8)
    for partner in (0, 1, -128, -127, 0x55, -86, 127, -1):
        check_pair_ops(cfg, b, c, [partner] * len(b))


def test_n8_float_assists_exhaustive():
    """FALIGN over every (e, E) and FNORM over every (e, m), on a page-2
    machine (n = 128); the channel word buf_c starts as random junk."""
    cfg = Config(s=32, m=32, n=128, N=8)
    h = FFNHarness(cfg)
    a, b = _all_pairs(8)
    for m in (0, 1, -1, 64, -65, 127, -128, 0x2A):
        want = [ref.falign(e, m, E, 8) for e, E in zip(a, b)]
        result, second, _ = h.run(OP_FALIGN, a, b, partner=[m] * len(a))
        _check("FALIGN e", result, [w[0] for w in want], a, b)
        _check("FALIGN m", second, [w[1] for w in want], a, b)
    rng = random.Random(8)
    junk = [rng.randint(-128, 127) for _ in a]
    want = [ref.fnorm(e, m, 8) for e, m in zip(a, b)]
    result, second, _ = h.run(OP_FNORM, a, junk, partner=b)
    _check("FNORM e", result, [w[0] for w in want], a, b)
    _check("FNORM m", second, [w[1] for w in want], a, b)


def test_n16_random():
    cfg = Config(s=32, m=8, n=64, N=16)
    rng = random.Random(16)
    lo, hi = -(1 << 15), (1 << 15) - 1
    edges = [lo, lo + 1, -2, -1, 0, 1, 2, hi - 1, hi, -300, -303]
    for chunk in range(6):
        b = [rng.randint(lo, hi) for _ in range(8192)]
        c = [rng.randint(lo, hi) for _ in range(8192)]
        if chunk == 0:
            b[:len(edges) ** 2] = [x for x in edges for _ in edges]
            c[:len(edges) ** 2] = [y for _ in edges for y in edges]
        if chunk == 1:   # LOGADD windows and MAX ties live near small differences
            c = [min(max(x + rng.randint(-40, 40), lo), hi) for x in b]
        if chunk == 2:   # every SHIFT amount, including the saturation edges
            c = [rng.randint(-20, 20) for _ in b]
        carry = [rng.randint(0, 1) for _ in range(8192)]
        check_binary_ops(cfg, b, c, carry)
        partner = [rng.randint(lo, hi) for _ in range(8192)]
        check_pair_ops(cfg, b, c, partner)


if __name__ == "__main__":
    import time
    for name in ("test_n8_binary_exhaustive", "test_n8_pair_exhaustive",
                 "test_n8_float_assists_exhaustive", "test_n16_random"):
        t = time.time()
        globals()[name]()
        print(f"[PASS] {name} ({time.time() - t:.1f}s)")
