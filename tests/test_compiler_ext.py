"""Test C arithmetic, builtins, arrays, and two-word fixed point.

Interpreter results are compared with Python references; selected programs
also compare the transformer and interpreter memory.
"""

import random
import torch

from strandedcpu.compiler import compile_c, fix_value
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.transformer import Config, Computer, init_state, read_memory, get_pc
import strandedcpu.reference as ref

N = 8
CFG = dict(s=32, m=96, n=1024, N=N)


def interp(src, **cfg):
    """Compile and interpret; returns (variables, fix variables, interpreter)."""
    cfg = {**CFG, **cfg}
    c, memory, cmds, meta = compile_c(src, **cfg)
    vm = ISAInterpreter(c.s, c.m, c.N, memory, cmds)
    vm.run()
    ints = {k: vm.memory[a] for k, a in meta['variables'].items()}
    fixes = {k: fix_value(vm.memory[a], vm.memory[a + 1], c.N) for k, a in meta['fix_variables'].items()}
    return ints, fixes, vm


def on_transformer(src, n, max_steps=5000, **cfg):
    """Run on the transformer and require the interpreter's exact memory."""
    cfg = {**CFG, **cfg, 'n': n}
    c, memory, cmds, meta = compile_c(src, **cfg)
    assert c.s + c.m + len(cmds) <= n, f"program needs n >= {c.s + c.m + len(cmds)}"
    vm = ISAInterpreter(c.s, c.m, c.N, memory, cmds)
    steps = vm.run()
    comp = Computer(c)
    X = init_state(c, memory, cmds)
    with torch.no_grad():
        for _ in range(max_steps):
            if get_pc(X, c) == 0:
                break
            X = comp.step(X)
    assert get_pc(X, c) == 0, "transformer did not halt"
    assert read_memory(X, c) == vm.memory
    return steps


# int: *, /, %, shifts

VALUES = [0, 1, -1, 2, -2, 3, 7, -7, 10, -10, 63, -64, 100, -100, 127, -128, 85, -86]


def c_div(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def test_int_mul_div_mod():
    for a in VALUES:
        for b in VALUES:
            src = f"int main() {{ int a = {a}; int b = {b}; int p = a * b; int q = 0; int r = 0;"
            if b != 0:
                src += " q = a / b; r = a % b;"
            src += " return p; }"
            v, _, _ = interp(src)
            assert v['p'] == ref.wrap(a * b, N), (a, b)
            if b != 0:
                q = ref.wrap(c_div(a, b), N)
                assert v['q'] == q, (a, b, v['q'], q)
                assert v['r'] == ref.wrap(a - b * c_div(a, b), N), (a, b)


def test_int_shifts():
    for x in (1, -1, 5, -100, 127, -128, 85):
        for k in range(N):
            src = f"int main() {{ int x = {x}; int k = {k}; int l = x << k; int r = x >> k; int lc = x << {k}; int rc = x >> {k}; return 0; }}"
            v, _, _ = interp(src)
            assert v['l'] == v['lc'] == ref.wrap(x << k, N), (x, k)
            assert v['r'] == v['rc'] == x >> k, (x, k)


def test_constant_folding_matches_runtime():
    src = "int main() { int a = -7 / 2; int b = -7 % 2; int c = 7 * -3; return 0; }"
    v, _, _ = interp(src)
    assert (v['a'], v['b'], v['c']) == (-3, -1, -21)


# Builtins for opcodes 21-31

def test_min_max_abs_exact():
    for a in VALUES:
        for b in VALUES:
            v, _, _ = interp(f"int main() {{ int a = {a}; int b = {b}; int lo = min(a, b); int hi = max(a, b); int m = abs(a); return 0; }}")
            assert (v['lo'], v['hi'], v['m']) == (min(a, b), max(a, b), ref.wrap(abs(a), N)), (a, b)


def test_logadd_clz_lfsr_umulhi_qdiv():
    rng = random.Random(1)
    for _ in range(60):
        a, b = rng.randint(-128, 127), rng.randint(-128, 127)
        ua, ub = rng.randint(0, 255), rng.randint(1, 255)
        src = f"""int main() {{ int a = {a}; int b = {b}; int ua = {ref.wrap(ua, N)}; int ub = {ref.wrap(ub, N)};
            int la = logadd(a, b); int z = clz(a); int nx = lfsr(a, b); int h = umulhi(ua, ub);
            int f = 0; if (ua < ub) {{ f = qdiv(ua, ub); }} return 0; }}"""
        v, _, _ = interp(src)
        assert v['la'] == ref.logadd(a, b, N, 2)
        assert v['z'] == ref.clz(a, N)
        assert v['nx'] == ref.lfsr(a, b, N)
        assert ref.unsigned(v['h'], N) == (ua * ub) >> N
        if ref.wrap(ua, N) < ref.wrap(ub, N) and ua < ub:
            assert ref.unsigned(v['f'], N) == (ua << N) // ub


def test_xlat_argmax_initializers():
    src = """
    int sq[16] = {0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, -112, -87, -60, -31};
    int w[8] = {3, -9, 27, 27, 5, -128, 0, 26};
    int main() {
        int i = 11;
        int a = xlat(sq, i);
        int b = xlat(sq, 3);
        int c = argmax(w);
        int d = argmax(w, 1);
        int e = argmax(sq, 2);
        return a;
    }
    """
    c, memory, cmds, meta = compile_c(src, **CFG)
    assert meta['variables']['sq'] % 16 == 0 and meta['variables']['w'] % 8 == 0
    v, _, _ = interp(src)
    assert (v['a'], v['b'], v['c'], v['d'], v['e']) == (121, 9, 2, 0, 3)


# fix: Q(N).(N) in two words

SCALE = 1 << N
LIMIT = 1 << (2 * N - 1)


def raw(x):
    return round(x * SCALE)


def wrap2(v):
    v &= (1 << 2 * N) - 1
    return v - (1 << 2 * N) if v >= LIMIT else v


def ref_mul(X, Y):
    mag = (abs(X) * abs(Y)) >> N
    return wrap2(-mag if (X < 0) != (Y < 0) else mag)


def ref_div(X, Y):
    mag = (abs(X) << N) // abs(Y)
    return wrap2(-mag if (X < 0) != (Y < 0) else mag)


def ref_div_int(X, k):
    mag = abs(X) // abs(k)
    return wrap2(-mag if (X < 0) != (k < 0) else mag)


def fix_cases(count, seed):
    rng = random.Random(seed)
    edges = [0, 1, -1, SCALE, -SCALE, SCALE // 2, -SCALE // 2, LIMIT - 1, -LIMIT, 3 * SCALE + 77, -5 * SCALE - 3]
    cases = [(x, y) for x in edges for y in edges[:6]]
    cases += [(rng.randint(-LIMIT, LIMIT - 1), rng.randint(-LIMIT, LIMIT - 1)) for _ in range(count)]
    cases += [(rng.randint(-4 * SCALE, 4 * SCALE), rng.randint(-4 * SCALE, 4 * SCALE)) for _ in range(count)]
    return cases


def test_fix_arithmetic():
    for X, Y in fix_cases(80, 2):
        x, y = X / SCALE, Y / SCALE
        src = f"""fix x = {x!r}; fix y = {y!r}; fix s; fix d; fix p; fix q; fix nx; int fl;
        int main() {{ s = x + y; d = x - y; p = x * y; nx = -x; fl = (int)x;"""
        if Y:
            src += " q = x / y;"
        src += " return 0; }"
        v, f, _ = interp(src)
        assert raw(f['s']) == wrap2(X + Y), (x, y)
        assert raw(f['d']) == wrap2(X - Y), (x, y)
        assert raw(f['p']) == ref_mul(X, Y), (x, y, f['p'])
        assert raw(f['nx']) == wrap2(-X), x
        assert v['fl'] == X >> N, x
        if Y:
            assert raw(f['q']) == ref_div(X, Y), (x, y, f['q'])


def test_fix_div_int_and_compare():
    rng = random.Random(3)
    for X, Y in fix_cases(40, 4):
        k = rng.choice([1, -1, 2, 3, -7, 100, -128, 127])
        x, y = X / SCALE, Y / SCALE
        src = f"""fix x = {x!r}; fix y = {y!r}; fix q; int lt; int gt; int le; int ge; int eq; int ne;
        int main() {{ q = x / {k}; lt = x < y; gt = x > y; le = x <= y; ge = x >= y; eq = x == y; ne = x != y;
        return 0; }}"""
        v, f, _ = interp(src)
        assert raw(f['q']) == ref_div_int(X, k), (x, k, f['q'])
        assert (v['lt'], v['gt'], v['le'], v['ge'], v['eq'], v['ne']) == \
            (int(X < Y), int(X > Y), int(X <= Y), int(X >= Y), int(X == Y), int(X != Y)), (x, y)


def test_fix_mixed_with_int_and_functions():
    src = """
    fix half(fix v) { return v / 2; }
    fix acc;
    int main() {
        int i;
        acc = 0.0;
        for (i = 1; i <= 4; i += 1) {
            acc += half(i * 1.5);
        }
        if (acc > 7) { acc -= 0.5; }
        if (acc) { i = 99; }
        return 0;
    }
    """
    v, f, _ = interp(src)
    assert f['acc'] == 7.0          # (1.5 + 3 + 4.5 + 6) / 2 = 7.5, minus 0.5
    assert v['i'] == 99


# Transformer agrees with the interpreter

def test_transformer_int_program():
    on_transformer("""
    int t[8] = {5, -3, 99, 12, 99, -128, 0, 7};
    int main() {
        int a = -100; int b = 7; int k = 3;
        int p = a * b; int q = a / b; int r = a % b;
        int s = a >> k; int u = b << k;
        int lo = min(a, b); int hi = max(a, b);
        int m = argmax(t); int x = xlat(t, k); int z = clz(b);
        int g = logadd(-40, -42); int h = umulhi(-56, 100);
        return p;
    }
    """, n=256, m=64)


def test_transformer_fix_program():
    on_transformer("""
    fix x = 1.5; fix y = -2.25; fix z; fix w; fix e; int c;
    int main() {
        z = x + y;
        w = x * y;
        e = w / 3;
        c = (w < z) + (z == -0.75);
        z = x / y;
        return c;
    }
    """, n=512, m=64, max_steps=3000)


def test_operands_survive_sibling_code():
    """Each case once returned a wrong value: the first operand sat in a
    scratch word (T0..T3, RETVAL, a fix temp) that the second's code reused,
    or (unary minus) was zeroed before being read."""
    head = ("int a; int b; int c; int i; int j; int r; int t[4]; fix x; fix y; fix z;\n"
            "int twice(int v) { return v + v; }\n"
            "int add3(int p, int q, int w) { return p + q + w; }\n"
            "fix fhalf(fix v) { fix h; h = v * 0.5; return h; }\n"
            "fix fclip(fix v) { if (v < 1.0) { return v; } return 1.0; }\n"
            "int main() { a = 5; b = 9; c = -3; i = 1; j = 2; t[2] = 7; x = 1.5; y = 2.0; ")
    cases = [
        ("r = -(b >> 1);", -4),
        ("r = -(a + b);", -14),
        ("r = twice(3) + twice(5);", 16),
        ("r = twice(3) - twice(5);", -4),
        ("r = min(twice(1), twice(2));", 2),
        ("r = add3(1, add3(2, 3, 4), 5);", 15),
        ("r = (a + 1 < b) + (c + 1 < a);", 2),
        ("r = b | logadd(6, a & c);", 9 | ref.logadd(6, 5 & -3, N, 2)),
        ("t[i] = t[j]; r = t[1];", 7),
        ("t[i] = b ^ (a == c); r = t[1];", 9),
        ("t[i] += t[j] * 2; r = t[1];", 14),
        ("z = x * fhalf(y); r = z * 4;", 6),
        ("z = (x + y) * (y - fhalf(x)); r = z * 4;", 17),
        ("z = 0.5; r = z < fclip(y);", 1),
    ]
    for body, want in cases:
        got = interp(head + body + " return 0; }")[0]['r']
        assert got == want, f"{body}: {got} != {want}"


def test_noinline_calls_on_both_sides():
    src = ("int r; noinline int twice(int v) { return v + v; }\n"
           "noinline int sq(int v) { return v * v; }\n"
           "int main() { r = twice(3) * 10 + sq(twice(2)) - twice(sq(2)); return 0; }")
    assert interp(src, m=96, n=1024)[0]['r'] == 60 + 16 - 8


def test_remainder_is_not_inexact():
    """int_div_remainder counts / that drops a fraction; % is exact."""
    from strandedcpu.api import score_options
    rem = score_options("int a; int b; int main() { return a % b; }", {'a': 17, 'b': 5}, {'A': '2'}, N=16)
    quo = score_options("int a; int b; int main() { return a / b; }", {'a': 17, 'b': 5}, {'A': '3'}, N=16)
    assert rem['status'] == quo['status'] == 'ok'
    assert rem['inexact_ops']['int_div_remainder'] == 0
    assert quo['inexact_ops']['int_div_remainder'] == 1


def test_return_leaves_the_function():
    """return used to fall through to the next statement in inline functions
    and main (only noinline bodies jumped to their epilogue)."""
    src = ("int r; int s; int q; int t[4] = {4, 8, 15, 16};\n"
           "int clamp(int v) { if (v < 0) { return 0; } return v; }\n"
           "int find(int v) { int k; for (k = 0; k < 4; k += 1) { if (t[k] == v) { return k; } } return -1; }\n"
           "int main() { r = clamp(-5); s = find(15); q = 1; return 0; q = 2; }")
    out = interp(src)[0]
    assert (out['r'], out['s'], out['q']) == (0, 2, 1)
    # a final return costs no jump
    _, _, cmds, _ = compile_c("int r; int f(int v) { return v + 1; }\nint main() { r = f(2); return 0; }", **CFG)
    assert sum(c[0] == 3 for c in cmds) == 0     # OP_JMP


def test_break_and_continue():
    src = ("int r; int s; int q; int i; int j;\n"
           "int first(int v) { int k; for (k = 0; k < 9; k += 1) { if (k * k >= v) { break; } } return k; }\n"
           "int main() { r = 0; for (i = 0; i < 10; i += 1) { if (i == 3) { continue; } if (i == 6) { break; } r += i; }\n"
           "  s = 0; i = 0; while (1) { i += 1; if (i % 2 == 0) { continue; } if (i > 9) { break; } s += i; }\n"
           "  q = 0; for (i = 0; i < 4; i += 1) { for (j = 0; j < 4; j += 1) { if (j > i) { break; } q += first(10); } }\n"
           "  return 0; }")
    out = interp(src)[0]
    assert (out['r'], out['s'], out['q']) == (12, 25, 40)
    for bad in ("int main() { break; return 0; }",
                "int f() { continue; return 0; }\nint main() { int i; for (i = 0; i < 2; i += 1) { f(); } return 0; }"):
        try:
            compile_c(bad, **CFG)
        except SyntaxError:
            continue
        raise AssertionError(f"accepted: {bad}")


def test_fast_lexer_matches_reference():
    """lex (one compiled regex) must give lex_reference's tokens exactly."""
    import strandedcpu.compiler as cc
    from tests import test_fuzz
    rng = random.Random(4)
    srcs = [cc.RAT_RUNTIME, "int a; int main() { a = 3-1; a = 3 -1; a = (-1); a = a--1; /* x\ny */ return -2; } // z",
            "int x; x = 1.5-0.25; x =-0.25;", "a = b /* unterminated", "int a; a = 1 @ 2;"]
    srcs += [test_fuzz.Gen(rng, 8, fix=k % 2 == 0).program() for k in range(60)]
    srcs += [test_fuzz.FixGen(rng).program() for _ in range(30)]
    for src in srcs:
        def tokens(f):
            try:
                return [(t.type, t.value, t.line) for t in f(src)]
            except SyntaxError as exc:
                return str(exc)
        assert tokens(cc.lex) == tokens(cc.lex_reference), src[:80]
