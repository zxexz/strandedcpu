"""Test sampling, inference, transforms, and numeric runtimes compiled from C.

Most programs use the interpreter. The Gibbs sampler, vector builtins, and
mailbox also compare transformer memory with the interpreter.
"""

import cmath
import math
import random
import statistics

import torch

import strandedcpu.reference as ref
import strandedcpu.runtime as rt
from strandedcpu.compiler import compile_c
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.api import run_function
from strandedcpu.host import run_with_mailbox
from strandedcpu.transformer import Computer, Config, init_state, read_memory, get_pc


def interp(src, inputs=None, N=16, s=128, m=256):
    return run_function(src, inputs, N=N, s=s, m=m)


def matches_transformer(src, N=8, s=64, m=64, max_steps=20000):
    cfg, memory, cmds, meta = compile_c(src, s=s, m=m, n=4096, N=N)
    n = 128
    while n < s + m + len(cmds):
        n *= 2
    vm = ISAInterpreter(s, m, N, memory, cmds)
    vm.run()
    tcfg = Config(s=s, m=m, n=n, N=N)
    comp = Computer(tcfg)
    X = init_state(tcfg, memory, cmds)
    with torch.no_grad():
        for _ in range(max_steps):
            if get_pc(X, tcfg) == 0:
                break
            X = comp.step(X)
    assert get_pc(X, tcfg) == 0
    assert read_memory(X, tcfg) == vm.memory
    return {k: vm.memory[a] for k, a in meta['variables'].items()}


# Gumbel-max categorical sampling (vector)

P = [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005]
LOGP = [math.floor(4 * math.log2(p) + 0.5) for p in P]


def gibbs_source(draws):
    return f"""
int hist[8]; int noise[8]; int logp[8] = {{{', '.join(map(str, LOGP))}}};
int seed; int i; int k; int v;
""" + rt.gumbel_table('gum', 8, 6) + f"""
int main() {{
    seed = 0;
    for (i = 0; i < {draws}; i += 1) {{
        vrand(noise, seed); seed += 1;
        vxlat(noise, gum);
        vadd(noise, logp);
        k = argmax(noise);
        v = hist[k]; hist[k] = v + 1;
    }}
    return k;
}}
"""


def gibbs_replica(draws, noise_base):
    gum = [math.floor(4 * -math.log(-math.log((k + 0.5) / 64)) / math.log(2) + 0.5) for k in range(64)]
    hist = [0] * 8
    for seed in range(draws):
        g = [gum[ref.vrand(seed, noise_base + i, 8) & 63] for i in range(8)]
        hist[ref.argmax([ref.wrap(a + b, 8) for a, b in zip(g, LOGP)])] += 1
    return hist


def test_gumbel_max_sampling():
    draws = 120                                   # i, seed are 8-bit words
    _, _, _, meta = compile_c(gibbs_source(draws), s=64, m=256, n=4096, N=8)
    out = interp(gibbs_source(draws), N=8, s=64, m=256)
    assert out['hist'] == gibbs_replica(draws, meta['variables']['noise'])
    q = [2 ** (lp / 4) for lp in LOGP]
    q = [x / sum(q) for x in q]
    tv = sum(abs(h / draws - p) for h, p in zip(out['hist'], q)) / 2
    assert tv < 0.15, (out["hist"], tv)


def test_gumbel_max_on_transformer():
    """Three draws, every memory word identical to the interpreter's."""
    matches_transformer(gibbs_source(3), N=8, s=64, m=256)


# Random-walk Metropolis (scalar)

def rwm_source(iters, seed):
    # target: N(10, 4^2); log2 pi in quarter bits = -(t-10)^2 * 4 / (32 ln 2) ~ -(t-10)^2 * 23 / 128
    return f"""
int hist[32]; int x; int xp; int lp; int lpp; int u; int lu; int k; int acc; int i; int v; int dlt; int q;
""" + rt.xorshift(16) + rt.log_domain(16) + rt.gaussian_table('gprop', 16, 6, 3.0) + f"""
int logpi(int t) {{ q = t - 10; q = q * q; q = q * 23; q = q >> 7; q = 0 - q; return q; }}
int main() {{
    {rt.xorshift_seed(seed, 16)}
    x = 10; acc = 0; lp = logpi(x);
    for (i = 0; i < {iters}; i += 1) {{
        u = rt_rand(); k = u & 63; dlt = xlat(gprop, k);
        xp = x + dlt;
        lpp = logpi(xp);
        u = rt_rand(); u = u | 1; lu = rt_log2q(u);
        dlt = lpp - lp;
        if (lu < dlt) {{ x = xp; lp = lpp; acc += 1; }}
        k = x + 6;
        if (k < 0) {{ k = 0; }}
        if (k > 31) {{ k = 31; }}
        v = hist[k]; hist[k] = v + 1;
    }}
    return x;
}}
"""


def test_random_walk_metropolis():
    iters = 1500
    out = interp(rwm_source(iters, 2463534242), N=16)
    hist = out['hist']
    assert sum(hist) == iters
    mean = sum((k - 6) * h for k, h in enumerate(hist)) / iters
    var = sum(((k - 6) - mean) ** 2 * h for k, h in enumerate(hist)) / iters
    assert abs(mean - 10) < 1.5 and 9 < var < 26, (mean, var)
    assert 0.3 < out['acc'] / iters < 0.95


# SVI over a stream of minibatches

SVI = rt.mailbox(8) + rt.xorshift(16) + rt.gaussian_table('gq', 16, 6, 64.0) + f"""
fix mq; fix z; fix g; fix eps; fix xi; int go; int b; int u; int k; int e;
fix loglik(fix mu, fix x) {{ return -(x - mu) * (x - mu) / 2; }}
fix logprior(fix mu) {{ return -(mu * mu) / 200; }}
int main() {{
    {rt.xorshift_seed(99991, 16)}
    mq = 0.0; go = 1;
    while (go) {{
        mb_wait();
        if (mb_len == 0) {{ go = 0; }} else {{
            u = rt_rand(); k = u & 63; e = xlat(gq, k);
            eps = (fix) e / 64;
            z = mq + eps / 2;                       // reparameterized draw, sigma_q = 1/2
            g = diff(logprior(z), z) / 4;           // prior gradient, shared over 4 batches
            for (b = 0; b < mb_len; b += 1) {{
                k = mb[b]; xi = k;
                g += diff(loglik(z, xi), z);
            }}
            mq += g / 32;                           // SGD, learning rate 2^-5
        }}
    }}
    return 0;
}}
"""


def svi_replay(batches, seed=99991):
    """SVI above in floats, on the same xorshift32 draws and Gaussian table."""
    nd = statistics.NormalDist()
    gq = [math.floor(64.0 * nd.inv_cdf((k + 0.5) / 64) + 0.5) for k in range(64)]
    state, mq = seed, 0.0
    for batch in batches:
        state = rt.xorshift32(state)
        z = mq + gq[state & 63] / 64 / 2
        g = -z / 100 / 4 + sum(x - z for x in batch)
        mq += g / 32
    return mq


def test_svi_with_mailbox():
    rng = random.Random(5)
    data = [round(rng.gauss(3.0, 1.0)) for _ in range(32)]
    batches = [data[i:i + 8] for i in range(0, 32, 8)] * 6         # 6 epochs
    out = run_with_mailbox(SVI, batches, N=16)
    # Compare with the floating-point replica; fix arithmetic rounds to 2^-16.
    assert abs(out['mq'] - svi_replay(batches)) < 1e-3, (out['mq'], svi_replay(batches))
    # Each step is mq <- 0.75 mq + 0.25 batch_mean - eps/8, so the estimate
    # keeps sum 0.75^j eps/8 of draw noise (sd ~0.19); with this seed it sits
    # 0.28 above the posterior mean for this seed.
    posterior_mean = sum(data) / (len(data) + 1 / 100)
    assert abs(out['mq'] - posterior_mean) < 0.6, (out['mq'], posterior_mean)


def test_mailbox_on_transformer():
    src = rt.mailbox(4) + """
int total; int go; int b; int v;
int main() {
    total = 0; go = 1;
    while (go) {
        mb_wait();
        if (mb_len == 0) { go = 0; } else {
            for (b = 0; b < mb_len; b += 1) { v = mb[b]; total += v; }
        }
    }
    return total;
}"""
    batches = [[3, -1, 4], [1, 5], [9, -2, 6, 5]]
    hw = run_with_mailbox(src, batches, N=8, s=32, m=32, backend="transformer")
    sw = run_with_mailbox(src, batches, N=8, s=32, m=32)
    assert hw['total'] == sw['total'] == sum(map(sum, batches))


# Transforms

def test_fft():
    N, F = 16, 12
    xs = [0.5, -0.25, 0.75, 0.1, -0.6, 0.3, 0.0, 0.9]
    out = interp(rt.fft(N, F, 3) + "int main() { fft_run(); return 0; }",
                 {'fft_re': [round(v * 4096) for v in xs], 'fft_im': [0] * 8})
    want = [sum(xs[j] * cmath.exp(-2j * math.pi * j * k / 8) for j in range(8)) / 8 for k in range(8)]
    err = max(abs(complex(out['fft_re'][k], out['fft_im'][k]) / 4096 - want[k]) for k in range(8))
    assert err < 4 / 4096


def test_ntt_polynomial_product():
    src = rt.ntt(8, 257, 4) + """int a[8]; int b[8]; int c[8]; int k; int v; int w;
int main() {
    for (k = 0; k < 8; k += 1) { v = a[k]; ntt_x[k] = v; }
    ntt_run(0);
    for (k = 0; k < 8; k += 1) { v = ntt_x[k]; c[k] = v; v = b[k]; ntt_x[k] = v; }
    ntt_run(0);
    for (k = 0; k < 8; k += 1) { v = ntt_x[k]; w = c[k]; v = mulmod(v, w, 257); ntt_x[k] = v; }
    ntt_run(1);
    return 0;
}"""
    A, B = [3, 1, 4, 1, 0, 0, 0, 0], [2, 7, 1, 8, 0, 0, 0, 0]
    out = interp(src, {'a': A, 'b': B})
    assert out['ntt_x'] == [sum(A[i] * B[k - i] for i in range(k + 1)) % 257 for k in range(8)]


def test_walsh_hadamard_builtin():
    vals = [3, -1, 4, 1, -5, 9, -2, 6]
    out = interp("int h[8]; int main() { wht(h); return 0; }", {'h': vals}, N=8, s=64, m=64)
    H = [[(-1) ** bin(i & j).count('1') for j in range(8)] for i in range(8)]
    assert out['h'] == [sum(H[i][j] * vals[j] for j in range(8)) for i in range(8)]


# Number formats

def test_lns():
    N, Z = 16, -(1 << 15)
    enc = lambda x: (0, Z) if x == 0 else (int(x < 0), round(math.log2(abs(x)) * 4))
    dec = lambda s, l: 0.0 if l == Z else (-1) ** s * 2 ** (l / 4)
    src = rt.lns(N) + """int sa; int la; int sb; int lb; int ms; int ml; int ds; int dl; int as_; int al; int sq;
int main() { lns_mul(sa, la, sb, lb); ms = lns_s; ml = lns_l; lns_div(sa, la, sb, lb); ds = lns_s; dl = lns_l;
             lns_add(sa, la, sb, lb); as_ = lns_s; al = lns_l; sq = lns_sqrt(la); return 0; }"""
    for x, y in [(3.0, 5.0), (-3.0, 5.0), (7.5, -7.0), (2.0, -2.0), (0.01, 100.0), (-0.5, -0.25), (0, 4.0), (6.0, 0)]:
        sa, la = enc(x)
        sb, lb = enc(y)
        o = interp(src, dict(sa=sa, la=la, sb=sb, lb=lb))
        xq, yq = dec(sa, la), dec(sb, lb)
        assert math.isclose(dec(o["ms"], o["ml"]), xq * yq)
        if yq:
            assert math.isclose(dec(o['ds'], o['dl']), xq / yq)
        s = xq + yq
        assert abs(dec(o['as_'], o['al']) - s) <= 0.1 * abs(s) + 1e-12, (x, y)
        if xq:
            assert abs(2 ** (o['sq'] / 4) - math.sqrt(abs(xq))) <= 0.1 * math.sqrt(abs(xq))


def test_block_floating_point():
    vals = [3, -7, 12, 0, 5, -1, 2, 9]
    out = interp(rt.bfp('blk', 8) + "int blk[8]; int main() { bfp_e = 0; bfp_normalize(); return 0; }",
                 {'blk': vals})
    assert [v * 2 ** out['bfp_e'] for v in out['blk']] == vals
    assert 1 << 14 <= max(abs(v) for v in out['blk']) < 1 << 15


def test_stochastic_rounding_is_unbiased():
    src = rt.xorshift(16) + rt.stochastic_rounding(16) + "int tot; int i; int v; int main() { " + \
        rt.xorshift_seed(12345, 16) + " tot = 0; for (i = 0; i < 400; i += 1) { v = rt_sround(5, 3); tot += v; } return 0; }"
    assert abs(interp(src)['tot'] / 400 - 5 / 8) < 0.05


def test_xorshift_matches_reference():
    for N in (8, 16):
        src = rt.xorshift(N) + "int out[8]; int i; int v; int main() { " + rt.xorshift_seed(2463534242, N) + \
            " for (i = 0; i < 8; i += 1) { v = rt_rand(); out[i] = v; } return 0; }"
        x, want = 2463534242, []
        for _ in range(8):
            x = rt.xorshift32(x)
            want.append(ref.wrap(x, N))
        assert interp(src, N=N)['out'] == want


def test_microcoded_float():
    src = rt.flt(16) + """int ax; int bx; int s; int p;
int main() {
    flt_from_int(ax); fa[0] = fr[0]; fa[1] = fr[1]; flt_from_int(bx); fb[0] = fr[0]; fb[1] = fr[1];
    flt_add(); s = flt_to_int();
    flt_from_int(ax); fa[0] = fr[0]; fa[1] = fr[1]; flt_from_int(bx); fb[0] = fr[0]; fb[1] = fr[1];
    flt_mul(); p = flt_to_int();
    return 0;
}"""
    for a, b in [(100, 23), (-50, 7), (3, -3), (1000, 30), (-120, -9), (5, 0), (0, 0)]:
        o = interp(src, dict(ax=a, bx=b))
        assert (o['s'], o['p']) == (a + b, a * b), (a, b)


# Compiler: noinline, mulmod, diff, vector builtins

def test_compiler_features():
    src = """
int a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
int b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int r[8];
int tot; int mx; int mm; int c1; int c2; int c3;
noinline int clamp(int v, int lo, int hi) {
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}
fix sq(fix t) { return t * t; }
fix f(fix u) { return sq(u - 2.0) * 3 + u / 4 + 1 / u; }
fix g; fix u0;
int main() {
    vadd(a, b);
    tot = vsum(a);
    vmov(r, b);
    mx = vmaxall(r);
    mm = mulmod(200, 300, 257);
    c1 = clamp(5, 0, 3); c2 = clamp(-9, 0, 3); c3 = clamp(2, 0, 3);
    u0 = 1.5;
    g = diff(f(u0), u0);
    return 0;
}"""
    o = interp(src)
    assert o['a'] == [396] * 8 and o['tot'] == 396 and o['mx'] == 80
    assert o['mm'] == 200 * 300 % 257 and (o['c1'], o['c2'], o['c3']) == (3, 0, 2)
    assert abs(o['g'] - (6 * (1.5 - 2) + 0.25 - 1 / 1.5 ** 2)) < 2 ** -12


def test_vector_builtins_on_transformer():
    v = matches_transformer("""
int a[8] = {1, -2, 3, 4, 50, 6, -7, 8};
int b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
int t; int x;
noinline int twice(int v) { return v + v; }
int main() { vadd(a, b); t = vmaxall(a); x = twice(t); return 0; }""", N=8, s=32, m=64)
    assert v['t'] == 100 and v['x'] == ref.wrap(200, 8)
