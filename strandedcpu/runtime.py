"""Generate C routines and word-size-specific lookup tables for the ISA.

Prepend the generators a program uses, for example cordic(8) + my_program.
Names use prefixes such as cr_ and rt_ because generated routines share the
compiler's flat memory namespace.

The module includes CORDIC, log-domain conversion, integer and fixed-point
routines, sampling tables, transforms and host-mailbox support.
CORDIC, Newton and FFT routines approximate real-valued calculations at word precision.
"""

import math
from typing import List, Sequence

from strandedcpu.transformer import logadd_table


def _table(name: str, values: Sequence[int], N: int, size: int = 0) -> str:
    """int name[size] = {...}; padded to a power of two (XLAT alignment)."""
    size = max(size, 1 << max(len(values) - 1, 0).bit_length())
    mask = (1 << N) - 1
    words = [((v & mask) - (1 << N) if (v & mask) >= 1 << (N - 1) else v & mask) for v in values]
    return f"int {name}[{size}] = {{{', '.join(map(str, words))}}};\n"


def cordic_frac_bits(N: int) -> int:
    """CORDIC words are Q2.(N-3): range [-4, 4)."""
    return N - 3


def cordic(N: int) -> str:
    """CORDIC in Q2.(N-3) words.

    cordic_sincos(theta)  sets cr_cos, cr_sin;   |theta| <= pi/2
    cordic_atan(y, x)     returns atan(y/x);     x > 0, |(x, y)| < 2.4 (gain 1.647)
    cordic_exp(t)         returns e^t;           |t| <= 1.1 (result < 4)
    cordic_ln(w)          returns ln w;          0.11 <= w < 3

    Each iteration rotates by +/-atan(2^-i) (circular) or +/-atanh(2^-i)
    (hyperbolic, with steps 4, 13, 40 repeated). The direction comes from a
    sign mask s = v >> (N-1), and cneg(u, s) = (u ^ s) - s applies it with no
    branch.
    """
    F = cordic_frac_bits(N)
    one = 1 << F
    q = lambda x: math.floor(x * one + 0.5)
    circ = list(range(F + 1))
    hyp, i = [], 1
    while i <= F:
        hyp.append(i)
        if i in (4, 13, 40):
            hyp.append(i)
        i += 1
    k_circ = math.prod(1 / math.sqrt(1 + 2.0 ** (-2 * i)) for i in circ)
    k_hyp = math.prod(math.sqrt(1 - 2.0 ** (-2 * i)) for i in hyp)
    src = _table("cr_atan", [q(math.atan(2.0 ** -i)) for i in circ], N)
    src += _table("cr_atanh", [q(math.atanh(2.0 ** -i)) for i in hyp], N)
    src += _table("cr_hshift", hyp, N)
    src += f"""int cr_cos; int cr_sin; int cr_x; int cr_y; int cr_z; int cr_i; int cr_k;
int cr_s; int cr_dx; int cr_dy; int cr_dz;

void cr_circular(int vectoring) {{
    for (cr_i = 0; cr_i < {len(circ)}; cr_i += 1) {{
        if (vectoring) {{ cr_s = cr_y >> {N - 1}; cr_s = cr_s ^ -1; }} else {{ cr_s = cr_z >> {N - 1}; }}
        cr_dx = cr_y >> cr_i; cr_dx = cr_dx ^ cr_s; cr_dx = cr_dx - cr_s;
        cr_dy = cr_x >> cr_i; cr_dy = cr_dy ^ cr_s; cr_dy = cr_dy - cr_s;
        cr_dz = xlat(cr_atan, cr_i); cr_dz = cr_dz ^ cr_s; cr_dz = cr_dz - cr_s;
        cr_x = cr_x - cr_dx; cr_y = cr_y + cr_dy; cr_z = cr_z - cr_dz;
    }}
}}

void cr_hyperbolic(int vectoring) {{
    for (cr_i = 0; cr_i < {len(hyp)}; cr_i += 1) {{
        cr_k = xlat(cr_hshift, cr_i);
        if (vectoring) {{ cr_s = cr_y >> {N - 1}; cr_s = cr_s ^ -1; }} else {{ cr_s = cr_z >> {N - 1}; }}
        cr_dx = cr_y >> cr_k; cr_dx = cr_dx ^ cr_s; cr_dx = cr_dx - cr_s;
        cr_dy = cr_x >> cr_k; cr_dy = cr_dy ^ cr_s; cr_dy = cr_dy - cr_s;
        cr_dz = xlat(cr_atanh, cr_i); cr_dz = cr_dz ^ cr_s; cr_dz = cr_dz - cr_s;
        cr_x = cr_x + cr_dx; cr_y = cr_y + cr_dy; cr_z = cr_z - cr_dz;
    }}
}}

void cordic_sincos(int theta) {{
    cr_x = {q(k_circ)}; cr_y = 0; cr_z = theta;
    cr_circular(0);
    cr_cos = cr_x; cr_sin = cr_y;
}}

int cordic_atan(int y, int x) {{
    cr_x = x; cr_y = y; cr_z = 0;
    cr_circular(1);
    return cr_z;
}}

int cordic_exp(int t) {{
    cr_x = {q(1 / k_hyp)}; cr_y = 0; cr_z = t;
    cr_hyperbolic(0);
    return cr_x + cr_y;
}}

int cordic_ln(int w) {{
    cr_x = w + {one}; cr_y = w - {one}; cr_z = 0;
    cr_hyperbolic(1);
    return cr_z + cr_z;
}}
"""
    return src


def log_domain(N: int, frac_bits: int = 2, table_bits: int = 3) -> str:
    """Conversions to and from LOGADD's units (log2 with frac_bits fraction bits).

    rt_log2(x)    log2 of an unsigned integer word x > 0
    rt_log2q(p)   log2 of a Q0.N probability p > 0 (always <= 0)
    rt_exp2q(v)   2^(v / 2^F) as a Q0.N probability, for log-probabilities v < 0

    CLZ finds the exponent, SHIFT normalizes the mantissa and XLAT looks up
    the fractional part.
    """
    F, T = frac_bits, table_bits
    log_t = [math.floor((1 << F) * math.log2(1 + i / (1 << T)) + 0.5) for i in range(1 << T)]
    # mantissa 2^(f / 2^F) in Q1.(N-2), so it stays positive as a signed word
    exp_t = [math.floor((1 << (N - 2)) * 2 ** (f / (1 << F)) + 0.5) for f in range(1 << F)]
    src = _table("rt_log2t", log_t, N) + _table("rt_exp2t", exp_t, N)
    src += f"""int rt_e; int rt_m; int rt_r;

int rt_log2(int x) {{
    rt_e = clz(x);
    rt_m = x << rt_e;
    rt_m = rt_m >> {N - 1 - T};
    rt_m = rt_m & {(1 << T) - 1};
    rt_r = {N - 1} - rt_e;
    rt_r = rt_r << {F};
    rt_r = rt_r + xlat(rt_log2t, rt_m);
    return rt_r;
}}

int rt_log2q(int p) {{
    return rt_log2(p) - {N << F};
}}

int rt_exp2q(int v) {{
    rt_e = v >> {F};
    rt_m = v & {(1 << F) - 1};
    rt_r = xlat(rt_exp2t, rt_m);
    rt_e = rt_e + 2;
    rt_r = rt_r << rt_e;
    return rt_r;
}}
"""
    return src


def newton(N: int) -> str:
    """Fixed-point reciprocal and square-root approximations by Newton iteration.

    rt_recip(d)   approximate 1/d for Q0.N d in (0.5, 1), as Q1.(N-1)
                  (d = 0.5 would be 2.0, which wraps to 0)
                  (so 1.0 = 2^(N-1)). Seed from an 8-entry XLAT table on the
                  bits after the leading one, then r = r * (2 - d*r) twice,
                  each product an umulhi (MULSTEP).
    rt_sqrtq(a)   approximate sqrt(a) for Q0.N a > 0, returned as Q0.N.
                  Run N updates of y = (y + qdiv(a,y)) / 2, starting from
                  the largest representable fraction, 1-2^-N. Quotients and
                  updates truncate to word precision.
    """
    seed = [math.floor((1 << (N - 1)) / ((8 + i + 0.5) / 16) + 0.5) for i in range(8)]
    src = _table("rt_rseed", seed, N)
    half = (1 << (N - 1)) - 1
    src += f"""int rt_nr; int rt_ne; int rt_ni; int rt_ny; int rt_nq; int rt_nh;

int rt_recip(int d) {{
    rt_ni = d >> {N - 4};
    rt_ni = rt_ni & 7;
    rt_nr = xlat(rt_rseed, rt_ni);
    for (rt_ni = 0; rt_ni < 2; rt_ni += 1) {{
        rt_ne = umulhi(d, rt_nr);
        rt_ne = 0 - rt_ne;
        rt_nr = umulhi(rt_nr, rt_ne);
        rt_nr = rt_nr << 1;
    }}
    return rt_nr;
}}

int rt_sqrtq(int a) {{
    rt_ny = -1;
    for (rt_ni = 0; rt_ni < {N}; rt_ni += 1) {{
        rt_nq = qdiv(a, rt_ny);
        rt_nh = rt_ny & rt_nq;
        rt_nh = rt_nh & 1;
        rt_ny = rt_ny >> 1;
        rt_ny = rt_ny & {half};
        rt_nq = rt_nq >> 1;
        rt_nq = rt_nq & {half};
        rt_ny = rt_ny + rt_nq;
        rt_ny = rt_ny + rt_nh;
    }}
    return rt_ny;
}}
"""
    return src


def integer_ops(N: int) -> str:
    """Generate popcount, GCD, saturating addition and integer square root.

    rt_popcount clears one set bit per iteration. rt_gcd accepts nonnegative
    inputs; rt_isqrt returns floor(sqrt(n)) for n >= 0. rt_sat_add clamps to the
    signed N-bit range.
    """
    top, bot = (1 << (N - 1)) - 1, -(1 << (N - 1))
    return f"""int rt_pc; int rt_pv; int rt_ga; int rt_gb; int rt_gt; int rt_ss; int rt_so; int rt_sx;
int rt_ix; int rt_iy; int rt_in;

int rt_popcount(int x) {{
    rt_pc = 0; rt_pv = x;
    while (rt_pv != 0) {{ rt_pc += 1; rt_gt = rt_pv - 1; rt_pv = rt_pv & rt_gt; }}
    return rt_pc;
}}

int rt_gcd(int a, int b) {{
    rt_ga = a; rt_gb = b;
    while (rt_gb != 0) {{ rt_gt = rt_ga % rt_gb; rt_ga = rt_gb; rt_gb = rt_gt; }}
    return rt_ga;
}}

int rt_sat_add(int a, int b) {{
    rt_ss = a + b;
    rt_so = a ^ rt_ss; rt_sx = b ^ rt_ss; rt_so = rt_so & rt_sx;
    if (rt_so < 0) {{
        if (a < 0) {{ rt_ss = {bot}; }} else {{ rt_ss = {top}; }}
    }}
    return rt_ss;
}}

int rt_isqrt(int n) {{
    rt_in = n; rt_ix = n;
    if (rt_in > 1) {{
        rt_iy = rt_ix / 2; rt_iy = rt_iy + 1;
        while (rt_iy < rt_ix) {{ rt_ix = rt_iy; rt_iy = rt_in / rt_ix; rt_iy = rt_iy + rt_ix; rt_iy = rt_iy / 2; }}
    }}
    return rt_ix;
}}
"""


def forward_differences(values: Sequence[int]) -> List[int]:
    """[p(0), Δp(0), Δ²p(0), ...] from the first degree+1 values of a polynomial."""
    diffs, row = [], list(values)
    while row:
        diffs.append(row[0])
        row = [b - a for a, b in zip(row, row[1:])]
    return diffs


def difference_engine(name: str, first_values: Sequence[int], count: int) -> str:
    """Babbage's method: fill int name[count] with a polynomial's values at
    evenly spaced points, given its first degree+1 values (exact integers).
    Each point costs one ADD per difference order and no multiplies.

    Call {name}_tabulate() once.
    """
    d = forward_differences(first_values)
    decl = "".join(f" int de_{name}_{k} = {v};" for k, v in enumerate(d))
    adds = "".join(f" de_{name}_{k} += de_{name}_{k + 1};" for k in range(len(d) - 1))
    return f"""int {name}[{count}]; int de_{name}_i;
void {name}_tabulate() {{{decl}
    for (de_{name}_i = 0; de_{name}_i < {count}; de_{name}_i += 1) {{
        {name}[de_{name}_i] = de_{name}_0;{adds}
    }}
}}
"""


def trapezoid(array: str, log2_intervals: int, log2_step: int) -> str:
    """rt_trap = h * (f0/2 + f1 + ... + f_{{n-1}} + fn/2) over int array[n + 1],
    n = 2^log2_intervals, h = 2^-log2_step. Call trapezoid_{array}()."""
    n = 1 << log2_intervals
    return f"""fix rt_trap; fix tz_{array}_e; int tz_{array}_i; int tz_{array}_v;
void trapezoid_{array}() {{
    rt_trap = 0.0;
    for (tz_{array}_i = 1; tz_{array}_i < {n}; tz_{array}_i += 1) {{ tz_{array}_v = {array}[tz_{array}_i]; rt_trap += tz_{array}_v; }}
    tz_{array}_e = {array}[0]; tz_{array}_e += {array}[{n}]; tz_{array}_e = tz_{array}_e / 2;
    rt_trap += tz_{array}_e;
    rt_trap = rt_trap / {1 << log2_step};
}}
"""


def simpson(array: str, log2_intervals: int, log2_step: int) -> str:
    """rt_simpson = (h/3) * (f0 + 4 f1 + 2 f2 + ... + 4 f_{{n-1}} + fn) over
    int array[n + 1], n = 2^log2_intervals, h = 2^-log2_step. The 4 and 2
    weights are doubling adds; one division by 3 * 2^log2_step finishes.
    Call simpson_{array}()."""
    n = 1 << log2_intervals
    return f"""fix rt_simpson; fix sp_{array}_odd; fix sp_{array}_even; int sp_{array}_i; int sp_{array}_v;
void simpson_{array}() {{
    sp_{array}_odd = 0.0; sp_{array}_even = 0.0;
    for (sp_{array}_i = 1; sp_{array}_i < {n}; sp_{array}_i += 2) {{ sp_{array}_v = {array}[sp_{array}_i]; sp_{array}_odd += sp_{array}_v; }}
    for (sp_{array}_i = 2; sp_{array}_i < {n}; sp_{array}_i += 2) {{ sp_{array}_v = {array}[sp_{array}_i]; sp_{array}_even += sp_{array}_v; }}
    sp_{array}_odd += sp_{array}_odd; sp_{array}_odd += sp_{array}_odd; sp_{array}_even += sp_{array}_even;
    rt_simpson = sp_{array}_odd + sp_{array}_even;
    rt_simpson += {array}[0]; rt_simpson += {array}[{n}];
    rt_simpson = rt_simpson / {3 << log2_step};
}}
"""


def logadd_units(frac_bits: int = 2) -> List[int]:
    """LOGADD's correction table, re-exported for programs that mirror it."""
    return logadd_table(frac_bits)


# Randomness, sampling tables, and number formats (page 2)

def xorshift32(x: int) -> int:
    """Marsaglia's xorshift32 (13, 17, 5), for reference."""
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


def _shift_words(src: List[str], dst: List[str], k: int, left: bool, N: int) -> str:
    """C that sets dst = src shifted by k bits as one W*N-bit number
    (words most significant first; right shifts are logical)."""
    W = len(src)
    q, r = divmod(k, N)
    out = []
    for i in range(W):
        parts = []
        hi = i + q if left else i - q              # word supplying the high part
        lo = hi + 1 if left else hi - 1            # neighbour supplying the spill
        if 0 <= hi < W:
            if left:
                parts.append(f"{src[hi]} << {r}" if r else src[hi])
            else:
                parts.append(f"rt_lsr({src[hi]}, {r})" if r else src[hi])
        if r and 0 <= lo < W:
            parts.append(f"rt_lsr({src[lo]}, {N - r})" if left else f"{src[lo]} << {N - r}")
        if not parts:
            out.append(f"{dst[i]} = 0;")
        else:
            out.append(f"{dst[i]} = {parts[0]};")
            for p in parts[1:]:
                out.append(f"rt_xt = {p}; {dst[i]} = {dst[i]} | rt_xt;")
    return " ".join(out)


def xorshift(N: int) -> str:
    """Generate xorshift32 over 32/N words, most significant word first.

    N must be a positive divisor of 32. rt_rand() returns the low word.
    At least one state word must be nonzero for the state period 2^32 - 1.
    """
    W = 32 // N
    s = [f"rs_{i}" for i in range(W)]
    t = [f"rt_xs{i}" for i in range(W)]
    decl = " ".join(f"int {n};" for n in s + t)
    lines = []
    for k, left in ((13, True), (17, False), (5, True)):
        lines.append(_shift_words(s, t, k, left, N))
        lines.append(" ".join(f"{a} = {a} ^ {b};" for a, b in zip(s, t)))
    body = "\n    ".join(lines)
    return f"""{decl} int rt_xt; int rt_lv;
int rt_lsr(int v, int k) {{
    rt_lv = v >> k;
    rt_lv = rt_lv & rt_lsr_mask(k);
    return rt_lv;
}}
int rt_lsr_mask(int k) {{
    rt_xt = 1 << k; rt_xt = rt_xt - 1; rt_xt = rt_xt << {N} - k; rt_xt = rt_xt ^ -1;
    return rt_xt;
}}
int rt_rand() {{
    {body}
    return {s[-1]};
}}
"""


def xorshift_seed(state: int, N: int) -> str:
    """C statements seeding rs_* with a 32-bit state."""
    W = 32 // N
    words = [(state >> (N * (W - 1 - i))) & ((1 << N) - 1) for i in range(W)]
    words = [w - (1 << N) if w >= 1 << (N - 1) else w for w in words]
    return " ".join(f"rs_{i} = {w};" for i, w in enumerate(words))


def gaussian_table(name: str, N: int, bits: int, scale: float) -> str:
    """Generate normal quantiles at (k+0.5)/2**bits, scaled and rounded to words.

    XLAT can index the table with uniform low bits. The finite table truncates
    the tails; at six index bits its extremes are about +/-2.4 standard deviations.
    """
    import statistics
    nd = statistics.NormalDist()
    size = 1 << bits
    return _table(name, [math.floor(scale * nd.inv_cdf((k + 0.5) / size) + 0.5) for k in range(size)], N)


def gumbel_table(name: str, N: int, bits: int, frac_bits: int = 2) -> str:
    """Generate quantized Gumbel noise in LOGADD's log2 units.

    Use frac_bits fractional bits and u=(k+0.5)/2**bits. Adding this noise to
    log2 probabilities approximates Gumbel-max sampling; finite table size and
    word rounding change the continuous distribution.
    """
    size = 1 << bits
    vals = [math.floor((1 << frac_bits) * -math.log(-math.log((k + 0.5) / size)) / math.log(2) + 0.5)
            for k in range(size)]
    return _table(name, vals, N)


def qmul(N: int, F: int, name: str = "rt_qmul") -> str:
    """name(a, b) = a * b >> F for signed QX.F words, truncated toward zero:
    the 2N-bit product of |a| and |b| from MULSTEP (umulhi and *), shifted
    across the word boundary. noinline: one copy, called through JMPI."""
    keep = (1 << (N - F)) - 1
    return f"""int {name}_s; int {name}_a; int {name}_b; int {name}_h; int {name}_l;
noinline int {name}(int a, int b) {{
    {name}_s = a ^ b; {name}_a = abs(a); {name}_b = abs(b);
    {name}_h = umulhi({name}_a, {name}_b); {name}_l = {name}_a * {name}_b;
    {name}_h = {name}_h << {N - F}; {name}_l = {name}_l >> {F}; {name}_l = {name}_l & {keep};
    {name}_h = {name}_h | {name}_l;
    if ({name}_s < 0) {{ {name}_h = 0 - {name}_h; }}
    return {name}_h;
}}
"""


def fft(N: int, F: int, log2n: int) -> str:
    """Generate an in-place radix-2 FFT on fft_re[n] and fft_im[n] QX.F words.

    n=2**log2n. VREV handles bit reversal; each butterfly uses four rt_qmul
    calls and halves its outputs, giving DFT(x)/n apart from fixed-point rounding.
    Inputs still need headroom for intermediate products and sums before that
    halving. Call fft_run().
    """
    n = 1 << log2n
    q = lambda v: math.floor(v * (1 << F) + 0.5)
    cos_t = [q(math.cos(2 * math.pi * k / n)) for k in range(n // 2)]
    sin_t = [q(-math.sin(2 * math.pi * k / n)) for k in range(n // 2)]
    return qmul(N, F) + _table("fft_cos", cos_t, N, size=n // 2) + _table("fft_sin", sin_t, N, size=n // 2) + f"""
int fft_re[{n}]; int fft_im[{n}]; int fft_tmp[{n}];
int ff_h; int ff_k; int ff_j; int ff_a; int ff_b; int ff_t; int ff_step;
int ff_wr; int ff_wi; int ff_xr; int ff_xi; int ff_tr; int ff_ti; int ff_ur; int ff_ui; int ff_v;
void fft_run() {{
    vrev(fft_tmp, fft_re); vmov(fft_re, fft_tmp);
    vrev(fft_tmp, fft_im); vmov(fft_im, fft_tmp);
    ff_step = {n // 2};
    for (ff_h = 1; ff_h < {n}; ff_h = ff_h + ff_h) {{
        for (ff_k = 0; ff_k < {n}; ff_k = ff_k + ff_h + ff_h) {{
            for (ff_j = 0; ff_j < ff_h; ff_j += 1) {{
                ff_t = ff_j * ff_step;
                ff_wr = fft_cos[ff_t]; ff_wi = fft_sin[ff_t];
                ff_a = ff_k + ff_j; ff_b = ff_a + ff_h;
                ff_xr = fft_re[ff_b]; ff_xi = fft_im[ff_b];
                ff_tr = rt_qmul(ff_xr, ff_wr); ff_v = rt_qmul(ff_xi, ff_wi); ff_tr = ff_tr - ff_v;
                ff_ti = rt_qmul(ff_xr, ff_wi); ff_v = rt_qmul(ff_xi, ff_wr); ff_ti = ff_ti + ff_v;
                ff_ur = fft_re[ff_a]; ff_ui = fft_im[ff_a];
                ff_v = ff_ur - ff_tr; fft_re[ff_b] = ff_v >> 1;
                ff_v = ff_ui - ff_ti; fft_im[ff_b] = ff_v >> 1;
                ff_v = ff_ur + ff_tr; fft_re[ff_a] = ff_v >> 1;
                ff_v = ff_ui + ff_ti; fft_im[ff_a] = ff_v >> 1;
            }}
        }}
        ff_step = ff_step >> 1;
    }}
}}
"""


def ntt(n: int, p: int, root: int) -> str:
    """Number-theoretic transform mod p (n-point, root of order n), O(n^2)
    with mulmod (MULSTEP product, DIVSTEP remainder): exact polynomial
    multiplication. ntt_run(dir): dir 0 forward, 1 inverse (scaled by 1/n),
    in place on int ntt_x[n]."""
    inv_root = pow(root, -1, p)
    inv_n = pow(n, -1, p)
    fw = [pow(root, k, p) for k in range(n)]
    bw = [pow(inv_root, k, p) for k in range(n)]
    return f"""int ntt_fw[{n}] = {{{', '.join(map(str, fw))}}};
int ntt_bw[{n}] = {{{', '.join(map(str, bw))}}};
int ntt_x[{n}]; int ntt_y[{n}];
int nt_k; int nt_j; int nt_e; int nt_acc; int nt_w; int nt_v;
void ntt_run(int dir) {{
    for (nt_k = 0; nt_k < {n}; nt_k += 1) {{
        nt_acc = 0; nt_e = 0;
        for (nt_j = 0; nt_j < {n}; nt_j += 1) {{
            if (dir) {{ nt_w = ntt_bw[nt_e]; }} else {{ nt_w = ntt_fw[nt_e]; }}
            nt_v = ntt_x[nt_j];
            nt_v = mulmod(nt_v, nt_w, {p});
            nt_acc = nt_acc + nt_v;
            if (nt_acc >= {p}) {{ nt_acc -= {p}; }}
            nt_e = nt_e + nt_k;
            if (nt_e >= {n}) {{ nt_e -= {n}; }}
        }}
        if (dir) {{ nt_acc = mulmod(nt_acc, {inv_n}, {p}); }}
        ntt_y[nt_k] = nt_acc;
    }}
    for (nt_k = 0; nt_k < {n}; nt_k += 1) {{ nt_v = ntt_y[nt_k]; ntt_x[nt_k] = nt_v; }}
}}
"""


def lns(N: int, frac_bits: int = 2) -> str:
    """Log number system: x = (-1)^sign * 2^(l / 2^F); zero has l = LNS_ZERO.
    Results land in lns_s, lns_l.

    lns_mul / lns_div   add / subtract the logs (XOR the signs)
    lns_sqrt(l)         l >> 1 (arithmetic shift)
    lns_add             same signs: LOGADD. Opposite signs: the larger
                        magnitude plus log2(1 - 2^-d) from a table, which
                        blows up as d -> 0 (exact cancellation gives zero).
    """
    F = frac_bits
    zero = -(1 << (N - 1))
    sub = [0] + [math.floor((1 << F) * math.log2(1 - 2 ** (-d / (1 << F))) + 0.5) for d in range(1, 32)]
    while sub and sub[-1] == 0 and len(sub) > 1:
        sub.pop()
    size = 1 << max(len(sub) - 1, 1).bit_length()
    sub += [0] * (size - len(sub))
    return _table("lns_sub", sub, N, size=size) + f"""int lns_s; int lns_l; int ln_d; int ln_v;
void lns_mul(int sa, int la, int sb, int lb) {{
    lns_s = sa ^ sb; lns_l = la + lb;
    if (la == {zero}) {{ lns_l = {zero}; }}
    if (lb == {zero}) {{ lns_l = {zero}; }}
}}
void lns_div(int sa, int la, int sb, int lb) {{
    lns_s = sa ^ sb; lns_l = la - lb;
    if (la == {zero}) {{ lns_l = {zero}; }}
}}
int lns_sqrt(int l) {{
    ln_v = l >> 1;
    return ln_v;
}}
void lns_add(int sa, int la, int sb, int lb) {{
    if (la >= lb) {{ lns_s = sa; lns_l = la; ln_d = la - lb; }} else {{ lns_s = sb; lns_l = lb; ln_d = lb - la; }}
    if (la == {zero}) {{ lns_s = sb; lns_l = lb; ln_d = -1; }}
    if (lb == {zero}) {{ lns_s = sa; lns_l = la; ln_d = -1; }}
    if (ln_d >= 0) {{
        if (sa == sb) {{
            lns_l = logadd(la, lb);
        }} else {{
            if (ln_d == 0) {{ lns_s = 0; lns_l = {zero}; }} else {{
                if (ln_d > {size - 1}) {{ ln_d = {size - 1}; }}
                ln_v = xlat(lns_sub, ln_d); lns_l = lns_l + ln_v;
            }}
        }}
    }}
}}
"""


def bfp(name: str, size: int) -> str:
    """Generate block normalization with shared exponent bfp_e.

    name holds size words, with size a power of two; value_i = name[i] * 2^bfp_e.
    For compiled word width N, inputs require |name[i]| < 2^(N-1) so negation
    represents their magnitudes. bfp_normalize() uses VMAX, CLZ and VSHIFT to
    move the largest magnitude's leading one to bit N-2, adjusting bfp_e.
    A zero block leaves the exponent unchanged.
    """
    return f"""int bfp_e; int bf_{name}_t[{size}]; int bf_{name}_z[{size}]; int bf_m; int bf_k;
void bfp_normalize() {{
    vmov(bf_{name}_t, {name}); vfill(bf_{name}_z, 0); vsub(bf_{name}_z, bf_{name}_t);
    vmax(bf_{name}_t, bf_{name}_z);
    bf_m = vmaxall(bf_{name}_t);
    if (bf_m != 0) {{
        bf_k = clz(bf_m); bf_k = bf_k - 1;
        vshift({name}, bf_k);
        bfp_e = bfp_e - bf_k;
    }}
}}
"""


def stochastic_rounding(N: int) -> str:
    """rt_sround(v, k): v / 2^k rounded at random with probability equal to
    the dropped fraction, so tiny updates survive on average. Needs
    xorshift(N) for rt_rand."""
    return f"""int sr_r; int sr_m; int sr_v;
int rt_sround(int v, int k) {{
    sr_m = 1 << k; sr_m = sr_m - 1;
    sr_r = rt_rand(); sr_r = sr_r & sr_m;
    sr_v = v + sr_r; sr_v = sr_v >> k;
    return sr_v;
}}
"""


def flt(N: int) -> str:
    """Generate arithmetic for two-word floats using FALIGN and FNORM.

    int f[2]={e,m} represents m*2**(e-(N-2)); normalized nonzero mantissas have
    different top two bits. fa and fb are clobbered; fr holds the result.
    flt_add aligns exponents and adds mantissas; flt_mul adds exponents and
    multiplies mantissas through rt_fmul. Both normalize the result.
    flt_from_int and flt_to_int convert integer values.
    """
    return qmul(N, N - 1, "rt_fmul") + f"""int fa[2]; int fb[2]; int fr[2]; int fl_t;
void flt_add() {{
    falign(fa, fb[0]); falign(fb, fa[0]);
    fl_t = fa[0] + 1; falign(fa, fl_t); falign(fb, fl_t);
    fr[0] = fa[0]; fl_t = fb[1]; fr[1] = fa[1] + fl_t;
    fnorm(fr);
}}
void flt_mul() {{
    fl_t = fb[0]; fr[0] = fa[0] + fl_t; fr[0] += 1;
    fr[1] = rt_fmul(fa[1], fb[1]);
    fnorm(fr);
}}
void flt_from_int(int x) {{
    fr[0] = {N - 2}; fr[1] = x; fnorm(fr);
}}
int flt_to_int() {{
    fl_t = fr[0] - {N - 2};
    fl_t = fr[1] << fl_t;
    return fl_t;
}}
"""


def mailbox(size: int) -> str:
    """Host mailbox: int mb[size], mb_len, mb_req. mb_wait() raises mb_req and
    spins until the host (strandedcpu.host.run_with_mailbox) has written the next
    batch into mb[0..mb_len-1] and cleared it; mb_len = 0 means no more data."""
    return f"""int mb_req; int mb_len; int mb[{size}];
void mb_wait() {{
    mb_req = 1;
    while (mb_req != 0) {{ }}
}}
"""
