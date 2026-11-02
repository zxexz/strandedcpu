"""Test numeric runtimes and log-domain HMM inference.

Compiled interpreter runs are compared with Python integer replicas and
approximate floating-point references. CORDIC and HMM forward also compare
transformer results.
"""

import itertools
import math
import torch

import strandedcpu.runtime as rt
import strandedcpu.reference as ref
from strandedcpu.compiler import compile_c, fix_value
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.transformer import Computer, init_state, read_memory, get_pc


def run(src, N=8, s=128, m=160):
    cfg, memory, cmds, meta = compile_c(src, s=s, m=m, n=4096, N=N)
    vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, memory, cmds)
    vm.run()
    out = {k: vm.memory[a] for k, a in meta['variables'].items()}
    out.update({k: fix_value(vm.memory[a], vm.memory[a + 1], N) for k, a in meta['fix_variables'].items()})
    return out, vm, meta


def run_on_transformer(src, n, N=8, s=32, m=64, max_steps=4000):
    cfg, memory, cmds, meta = compile_c(src, s=s, m=m, n=n, N=N)
    assert cfg.s + cfg.m + len(cmds) <= n
    vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, memory, cmds)
    vm.run()
    comp = Computer(cfg)
    X = init_state(cfg, memory, cmds)
    with torch.no_grad():
        for _ in range(max_steps):
            if get_pc(X, cfg) == 0:
                break
            X = comp.step(X)
    assert get_pc(X, cfg) == 0
    assert read_memory(X, cfg) == vm.memory
    return {k: vm.memory[a] for k, a in meta['variables'].items()}


# CORDIC

def test_cordic():
    for N, lsb_tol in ((8, 4), (16, 6)):
        one = 1 << rt.cordic_frac_bits(N)
        tol = lsb_tol / one
        lib = rt.cordic(N)
        for th in (-1.5, -0.7, 0.0, 0.3, 1.0, 1.55):
            v, _, _ = run(lib + f"int c; int s; int main() {{ cordic_sincos({round(th * one)}); c = cr_cos; s = cr_sin; return 0; }}", N)
            assert abs(v['c'] / one - math.cos(th)) < tol and abs(v['s'] / one - math.sin(th)) < tol, (N, th)
        for y, x in ((1, 2), (-1.5, 1), (0, 1), (1, 1)):
            v, _, _ = run(lib + f"int a; int main() {{ a = cordic_atan({round(y * one)}, {round(x * one)}); return 0; }}", N)
            assert abs(v['a'] / one - math.atan2(y, x)) < tol, (N, y, x)
        for t in (-1.0, -0.3, 0.0, 0.5, 1.0):
            v, _, _ = run(lib + f"int a; int main() {{ a = cordic_exp({round(t * one)}); return 0; }}", N)
            assert abs(v['a'] / one - math.exp(t)) < tol * 2, (N, t)
        for w in (0.25, 0.5, 1.0, 1.5, 2.5):
            v, _, _ = run(lib + f"int a; int main() {{ a = cordic_ln({round(w * one)}); return 0; }}", N)
            assert abs(v['a'] / one - math.log(w)) < tol, (N, w)


def test_cordic_on_transformer():
    one = 1 << rt.cordic_frac_bits(8)
    v = run_on_transformer(rt.cordic(8) + f"int c; int s; int main() {{ cordic_sincos({round(0.6 * one)}); c = cr_cos; s = cr_sin; return 0; }}",
                           n=256, s=32, m=64)
    assert abs(v['c'] / one - math.cos(0.6)) < 4 / one and abs(v['s'] / one - math.sin(0.6)) < 4 / one


# log2 / exp2, Newton, integer ops

def test_log_domain_conversions():
    for N in (8, 16):
        lib = rt.log_domain(N)
        for x in [1, 2, 3, 5, 7, 100, 127] + ([255, 1000, 30000] if N == 16 else []):
            v, _, _ = run(lib + f"int a; int main() {{ a = rt_log2({x}); return 0; }}", N)
            assert abs(v['a'] / 4 - math.log2(x)) <= 0.25, (N, x)
        for p in (0.5, 0.25, 0.1, 0.9, 0.01):
            raw = round(p * (1 << N))
            v, _, _ = run(lib + f"int a; int b; int main() {{ a = rt_log2q({ref.wrap(raw, N)}); b = rt_exp2q(a); return 0; }}", N)
            lp = v['a'] / 4
            assert abs(lp - math.log2(raw / (1 << N))) <= 0.25, (N, p)
            assert abs(ref.unsigned(v['b'], N) / (1 << N) - 2 ** lp) <= 2 / (1 << N) + 0.02 * 2 ** lp, (N, p)


def test_newton():
    for N in (8, 16):
        lib = rt.newton(N)
        for d in range((1 << (N - 1)) + 1, 1 << N, 7 if N == 8 else 997):
            v, _, _ = run(lib + f"int r; int main() {{ r = rt_recip({ref.wrap(d, N)}); return 0; }}", N)
            assert abs(ref.unsigned(v['r'], N) - (1 << (2 * N - 1)) / d) <= 3, (N, d)
        for a in ([1, 5, 50, 100, 200, 255] if N == 8 else [1, 100, 5000, 40000, 65535]):
            v, _, _ = run(lib + f"int r; int main() {{ r = rt_sqrtq({ref.wrap(a, N)}); return 0; }}", N)
            assert abs(ref.unsigned(v['r'], N) - math.sqrt(a * (1 << N))) <= 2, (N, a)


def test_integer_ops():
    for N in (8, 16):
        lim = 1 << (N - 1)
        for a, b in [(0, 0), (12, 18), (127, 1), (-1, 5), (100, 100), (-128, -1), (64, 64), (35, 49)]:
            ga, gb = abs(a) % 127, abs(b) % 127
            v, _, _ = run(rt.integer_ops(N) + f"int pc; int g; int sa; int q; int main() {{ pc = rt_popcount({a}); g = rt_gcd({ga}, {gb}); sa = rt_sat_add({a}, {b}); q = rt_isqrt({ga}); return 0; }}", N)
            assert v['pc'] == bin(ref.unsigned(a, N)).count('1')
            assert v['g'] == math.gcd(ga, gb)
            assert v['sa'] == max(-lim, min(lim - 1, a + b))
            assert v['q'] == math.isqrt(ga)


# Difference engine and integration

def test_difference_engine():
    poly = lambda x: 3 * x * x - 5 * x + 2
    src = rt.difference_engine('tab', [poly(x) for x in range(3)], 20) + "int main() { tab_tabulate(); return 0; }"
    _, vm, meta = run(src, N=16)
    base = meta['variables']['tab']
    assert vm.memory[base:base + 20] == [poly(x) for x in range(20)]
    # cubic, 8-bit: wraps exactly like the true values would
    cubic = lambda x: x ** 3 - 4 * x
    src = rt.difference_engine('cb', [cubic(x) for x in range(4)], 12) + "int main() { cb_tabulate(); return 0; }"
    _, vm, meta = run(src, N=8)
    base = meta['variables']['cb']
    assert vm.memory[base:base + 12] == [ref.wrap(cubic(x), 8) for x in range(12)]


def test_trapezoid_simpson():
    f = [round(100 * math.sin(math.pi * i / 16)) for i in range(17)]
    src = ("int f[17] = {" + ", ".join(map(str, f)) + "};\n" + rt.trapezoid('f', 4, 4) + rt.simpson('f', 4, 4)
           + "fix t; fix sp; int main() { trapezoid_f(); simpson_f(); t = rt_trap; sp = rt_simpson; return 0; }")
    v, _, _ = run(src, N=16)
    h = 1 / 16
    trap = h * (sum(f) - (f[0] + f[16]) / 2)
    simp = h / 3 * (f[0] + f[16] + 4 * sum(f[1:16:2]) + 2 * sum(f[2:15:2]))
    assert abs(v['t'] - trap) < 2 ** -16 and abs(v['sp'] - simp) < 2 ** -16
    assert abs(simp - 200 / math.pi) < 0.2       # integral of 100 sin(pi x) on [0, 1]


# Log-domain inference: HMM forward (LOGADD) and Viterbi (MAX + ARGMAX)
# Probabilities are stored as log2 p in quarter-bit units (log_frac_bits = 2).

F = 2
INIT = [0.6, 0.4]
TRANS = [[0.7, 0.3], [0.4, 0.6]]
EMIT = [[0.5, 0.4, 0.1, 0.0001], [0.1, 0.3, 0.6, 0.0001]]
OBS = [0, 1, 2, 2, 1]


def lq(p):
    return math.floor(math.log2(p) * (1 << F) + 0.5)


def hmm_source():
    init = ", ".join(str(lq(p)) for p in INIT)
    trans = ", ".join(str(lq(p)) for row in TRANS for p in row)
    emit = ", ".join(str(lq(p)) for row in EMIT for p in row)
    obs = ", ".join(map(str, OBS))
    return f"""
    int li[2] = {{{init}}};
    int lt[4] = {{{trans}}};      // lt[2*from + to]
    int le[8] = {{{emit}}};       // le[4*state + symbol]
    int obs[8] = {{{obs}}};
    int a0; int a1; int n0; int n1; int t; int o; int x; int y; int ll;
    int d0; int d1; int m0; int m1; int bp[16]; int cand[2]; int path[8]; int k; int s;

    int main() {{
        // forward: alpha_t(j) = logsum_i alpha_{{t-1}}(i) + lt[i][j], plus le[j][o_t]
        o = obs[0];
        a0 = li[0] + xlat(le, o);
        x = o + 4; a1 = li[1] + xlat(le, x);
        for (t = 1; t < {len(OBS)}; t += 1) {{
            o = obs[t];
            n0 = a0 + lt[0]; x = a1 + lt[2]; n0 = logadd(n0, x); n0 += xlat(le, o);
            n1 = a0 + lt[1]; x = a1 + lt[3]; n1 = logadd(n1, x); x = o + 4; n1 += xlat(le, x);
            a0 = n0; a1 = n1;
        }}
        ll = logadd(a0, a1);

        // Viterbi: delta_t(j) = max_i delta_{{t-1}}(i) + lt[i][j], backpointer = argmax
        o = obs[0];
        d0 = li[0] + xlat(le, o);
        x = o + 4; d1 = li[1] + xlat(le, x);
        for (t = 1; t < {len(OBS)}; t += 1) {{
            o = obs[t];
            cand[0] = d0 + lt[0]; cand[1] = d1 + lt[2];
            k = 2 * t; bp[k] = argmax(cand); m0 = max(cand[0], cand[1]); m0 += xlat(le, o);
            cand[0] = d0 + lt[1]; cand[1] = d1 + lt[3];
            k = k + 1; bp[k] = argmax(cand); m1 = max(cand[0], cand[1]); x = o + 4; m1 += xlat(le, x);
            d0 = m0; d1 = m1;
        }}
        cand[0] = d0; cand[1] = d1;
        s = argmax(cand);
        for (t = {len(OBS) - 1}; t >= 0; t -= 1) {{
            path[t] = s;
            k = 2 * t; k = k + s; s = bp[k];
        }}
        return ll;
    }}
    """


def hmm_reference():
    """The same integer program in Python, plus float log-likelihood and path."""
    N = 8
    li = [lq(p) for p in INIT]
    lt = [[lq(p) for p in row] for row in TRANS]
    le = [[lq(p) for p in row] for row in EMIT]
    w = lambda v: ref.wrap(v, N)
    a = [w(li[j] + le[j][OBS[0]]) for j in range(2)]
    for o in OBS[1:]:
        a = [w(ref.logadd(w(a[0] + lt[0][j]), w(a[1] + lt[1][j]), N, F) + le[j][o]) for j in range(2)]
    ll = ref.logadd(a[0], a[1], N, F)
    d = [w(li[j] + le[j][OBS[0]]) for j in range(2)]
    bps = []
    for o in OBS[1:]:
        cands = [[w(d[i] + lt[i][j]) for i in range(2)] for j in range(2)]
        bps.append([ref.argmax(c) for c in cands])
        d = [w(max(cands[j]) + le[j][o]) for j in range(2)]
    s = ref.argmax(d)
    path = [s]
    for bp in reversed(bps):
        s = bp[s]
        path.append(s)
    path.reverse()
    # float truth
    fa = [INIT[j] * EMIT[j][OBS[0]] for j in range(2)]
    for o in OBS[1:]:
        fa = [sum(fa[i] * TRANS[i][j] for i in range(2)) * EMIT[j][o] for j in range(2)]
    return ll, path, math.log2(sum(fa))


def test_hmm_forward_and_viterbi():
    v, vm, meta = run(hmm_source())
    ll, path, true_ll = hmm_reference()
    assert v['ll'] == ll
    base = meta['variables']['path']
    assert vm.memory[base:base + len(OBS)] == path
    assert abs(ll / (1 << F) - true_ll) < 0.75        # quarter-bit units, 5 steps of rounding
    # Quantized log-probabilities can turn a near-tie (here 0.16 vs 0.18 at the
    # last step) into an exact tie, so hold the decoded path to the float
    # optimum's probability rather than to its states.
    def path_logp(p):
        lp = math.log2(INIT[p[0]] * EMIT[p[0]][OBS[0]])
        return lp + sum(math.log2(TRANS[p[i - 1]][p[i]] * EMIT[p[i]][OBS[i]]) for i in range(1, len(p)))
    best = max(path_logp(p) for p in itertools.product((0, 1), repeat=len(OBS)))
    assert path_logp(path) > best - 0.5


def test_hmm_on_transformer():
    v = run_on_transformer(hmm_source(), n=512, s=128, m=96, max_steps=1500)
    assert v['ll'] == hmm_reference()[0]
