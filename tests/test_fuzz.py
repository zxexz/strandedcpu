"""Differential tests for the compiler, interpreter, and transformer.

Gen emits bounded C programs for both execution backends. CRef evaluates
the integer AST independently; FixRef uses Fractions with a tolerance for
binary fixed-point rounding. Binary cases near branch/cast ties are skipped.

STRANDEDCPU_FUZZ sets the random corpus size (default 24).
"""

import os
import random

import torch

import strandedcpu.reference as ref
import strandedcpu.compiler as cc
from fractions import Fraction as Fr

from strandedcpu.compiler import compile_c, fix_value, ExecutionFault
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.transformer import Config, Computer, init_state, read_memory, get_pc

PROGRAMS = int(os.environ.get('STRANDEDCPU_FUZZ', '24'))
_machines = {}


class Gen:
    """Random but well-defined programs: no division by zero, bounded loops,
    in-range array indexes."""

    def __init__(self, rng, N, fix=True):
        self.rng, self.N, self.fix = rng, N, fix
        self.ints = ['a', 'b', 'c', 'd']
        self.lim = 1 << (N - 1)

    def lit(self):
        return str(self.rng.randint(-self.lim // 4, self.lim // 4))

    def atom(self):
        r = self.rng.random()
        if r < 0.35:
            return self.lit()
        if r < 0.85:
            return self.rng.choice(self.ints)
        return f"t[{self.rng.randint(0, 7)}]"

    def expr(self, depth=0):
        if depth > 2 or self.rng.random() < 0.3:
            return self.atom()
        ops = ['+', '-', '*', '&', '|', '^', '<<', '>>', '/', '%', 'cmp', 'call', 'neg']
        if not self.fix:    # C-semantics programs (interpreter only) cover more of the language
            ops += ['logic', 'not', 'vshift', 'vdiv', 'clamp', 'find']
        op = self.rng.choice(ops)
        x, y = self.expr(depth + 1), self.expr(depth + 1)
        if op == 'logic':
            return f"({x} {self.rng.choice(['&&', '||'])} {y})"
        if op == 'not':
            return f"({self.rng.choice(['!', '~'])}{x})"
        if op == 'vshift':      # variable amount, always in 0..N-1
            return f"({x} {self.rng.choice(['<<', '>>'])} ({y} & {self.N - 1}))"
        if op == 'vdiv':        # variable divisor, never zero
            return f"({x} {self.rng.choice(['/', '%'])} ({y} | 1))"
        if op in ('clamp', 'find'):     # inline functions with early returns
            return f"{op}({x})"
        if op in ('<<', '>>'):
            return f"({x} {op} {self.rng.randint(0, self.N - 1)})"
        if op in ('/', '%'):
            return f"({x} {op} {self.rng.choice(['3', '7', '-5', '11'])})"
        if op == 'cmp':
            return f"({x} {self.rng.choice(['<', '<=', '>', '>=', '==', '!='])} {y})"
        if op == 'call':
            f = self.rng.choice(['max', 'min', 'abs', 'clz', 'logadd', 'twice'])
            return f"{f}({x})" if f in ('abs', 'clz', 'twice') else f"{f}({x}, {y})"
        if op == 'neg':
            return f"(-{x})"
        return f"({x} {op} {y})"

    def stmt(self, depth=0):
        r = self.rng.random()
        v = self.rng.choice(self.ints)
        if r < 0.45:
            return f"{v} = {self.expr()};"
        if r < 0.55:
            return f"{v} {self.rng.choice(['+=', '-='])} {self.expr()};"
        if r < 0.65:
            return f"t[{self.rng.choice(['i', str(self.rng.randint(0, 7))])}] = {self.expr()};"
        if r < 0.72:
            return f"{v} = t[i];"
        if r < 0.85 and depth < 2:
            body = " ".join(self.stmt(depth + 1) for _ in range(2))
            other = " ".join(self.stmt(depth + 1) for _ in range(1))
            return f"if ({self.expr(1)}) {{ {body} }} else {{ {other} }}"
        if depth < 1:
            body = " ".join(self.stmt(depth + 1) for _ in range(2))
            return f"for (i = 0; i < {self.rng.randint(1, 4)}; i += 1) {{ {body} }}"
        if not self.fix:
            if r < 0.93 or depth:       # one level: every while shares the counter s
                return f"{v} = {self.expr()};"
            body = " ".join(self.stmt(depth + 1) for _ in range(2))
            return f"s = 0; while (s < {self.rng.randint(1, 3)}) {{ {body} s += 1; }}"
        return f"x = x * {self.rng.choice(['1.5', '-0.25', '2.0'])} + (fix) {v};"

    def program(self):
        body = " ".join(self.stmt() for _ in range(self.rng.randint(4, 9)))
        vec = ("vadd(t, u); s = vsum(t); s = vmaxall(u);" if self.fix and self.rng.random() < 0.5 else "")
        fix_decl, fix_init = (" fix x;", " x = 0.75;") if self.fix else ("", "")
        return (f"int a; int b; int c; int d; int i; int s; int t[8]; int u[8] = {{1, 2, 3, 4, 5, 6, 7, 8}};{fix_decl}\n"
                f"noinline int twice(int v) {{ return v + v; }}\n{'' if self.fix else self.HELPERS}"
                f"int main() {{ a = {self.lit()}; b = {self.lit()}; c = {self.lit()}; d = {self.lit()}; i = 0;{fix_init}\n"
                f"  {body}\n  i = i & 7; {vec}\n  return a; }}\n")


Gen.HELPERS = ("int clamp(int v) { if (v < -9) { return -9; } if (v > 9) { return 9; } return v; }\n"
               "int find(int v) { int k; for (k = 0; k < 8; k += 1) { if (t[k] == v) { return k; } } return -1; }\n")


class _Return(Exception):
    pass


class CRef:
    """Evaluates the int subset Gen(fix=False) emits, straight from the AST,
    with C semantics on N-bit words. Globals only (as Gen writes them)."""

    def __init__(self, src, N):
        self.N = N
        prog = cc.Parser(cc.lex(src)).parse_program()
        self.funcs = {f.name: f for f in prog.functions}
        self.budget = 10 ** 6       # statements; RuntimeError past it
        self.g = {}
        for d in prog.globals:
            self.g[d.name] = ([ref.wrap(v, N) for v in (d.values or [])] + [0] * (d.size - len(d.values or []))
                              if d.size else 0)

    def run(self):
        try:
            self.block(self.funcs['main'].body)
        except _Return:
            pass
        return self.g

    def w(self, v):
        return ref.wrap(v, self.N)

    def ev(self, e):
        if isinstance(e, cc.IntLiteral):
            return self.w(e.value)
        if isinstance(e, cc.VarRef):
            return self.g[e.name]
        if isinstance(e, cc.ArrayRef):
            return self.g[e.name][self.ev(e.index)]
        if isinstance(e, cc.UnaryOp):
            v = self.ev(e.expr)
            return {'-': self.w(-v), '~': self.w(~v), '!': int(v == 0)}[e.op]
        if isinstance(e, cc.BinaryOp):
            if e.op in ('&&', '||'):
                x = self.ev(e.left) != 0
                return int(x and self.ev(e.right) != 0) if e.op == '&&' else int(x or self.ev(e.right) != 0)
            a, b = self.ev(e.left), self.ev(e.right)
            if e.op in ('/', '%'):
                q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
                return self.w(q if e.op == '/' else a - b * q)
            if e.op in ('<<', '>>'):
                return self.w(a << b if e.op == '<<' else a >> b)
            return self.w({'+': a + b, '-': a - b, '*': a * b, '&': a & b, '|': a | b, '^': a ^ b,
                           '<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b, '==': a == b, '!=': a != b}[e.op])
        if isinstance(e, cc.FuncCall):
            args = [self.ev(x) for x in e.args]
            builtin = {'max': max, 'min': min, 'abs': lambda v: self.w(abs(v)),
                       'clz': lambda v: ref.clz(v, self.N), 'logadd': lambda x, y: ref.logadd(x, y, self.N, 2)}
            if e.name in builtin:
                return builtin[e.name](*args)
            f = self.funcs[e.name]
            self.g.update(zip(f.params, args))
            try:
                self.block(f.body)
            except _Return as r:
                return r.args[0]
            return 0
        raise NotImplementedError(type(e).__name__)

    def block(self, b):
        for st in b.stmts:
            self.stmt(st)

    def stmt(self, st):
        self.budget -= 1
        if self.budget < 0:
            raise RuntimeError("CRef: step budget exhausted")
        if isinstance(st, cc.Assignment):
            v = self.ev(st.value)
            t = st.target
            box, key = (self.g, t.name) if isinstance(t, cc.VarRef) else (self.g[t.name], self.ev(t.index))
            box[key] = self.w(v if st.op == '=' else box[key] + v if st.op == '+=' else box[key] - v)
        elif isinstance(st, cc.IfStmt):
            if self.ev(st.cond):
                self.block(st.then_body)
            elif st.else_body:
                self.block(st.else_body)
        elif isinstance(st, cc.ForStmt):
            self.stmt(st.init)
            while self.ev(st.cond):
                self.block(st.body)
                self.stmt(st.update)
        elif isinstance(st, cc.WhileStmt):
            while self.ev(st.cond):
                self.block(st.body)
        elif isinstance(st, cc.VarDecl):
            self.g[st.name] = self.ev(st.init) if st.init else self.g.get(st.name, 0)
        elif isinstance(st, cc.ReturnStmt):
            raise _Return(self.ev(st.value) if st.value else 0)
        else:
            raise NotImplementedError(type(st).__name__)


def machine(s, m, n, N):
    key = (s, m, n, N)
    if key not in _machines:
        cfg = Config(s=s, m=m, n=n, N=N)
        _machines[key] = (cfg, Computer(cfg))
    return _machines[key]


def run_both(src, N=8, s=32, m=96):
    cfg0, memory, cmds, _ = compile_c(src, s=s, m=m, n=4096, N=N)
    n = 128
    while n < s + m + len(cmds):
        n *= 2
    vm = ISAInterpreter(s, m, N, memory, cmds)
    steps = vm.run(max_steps=20000)
    cfg, comp = machine(s, m, n, N)
    X = init_state(cfg, memory, cmds)
    with torch.no_grad():
        for _ in range(steps + 5):
            if get_pc(X, cfg) == 0:
                break
            X = comp.step(X)
    return read_memory(X, cfg), vm.memory, get_pc(X, cfg), steps


def test_random_programs_agree():
    rng = random.Random(2026)
    failures = []
    skipped = 0
    for k in range(PROGRAMS):
        src = Gen(rng, 8).program()
        try:
            hw, sw, pc, steps = run_both(src)
        except (SyntaxError, MemoryError, RuntimeError):     # generator produced something uncompilable
            skipped += 1
            continue
        if hw != sw or pc != 0:
            failures.append((k, src, [(i, a, b) for i, (a, b) in enumerate(zip(hw, sw)) if a != b][:5], pc))
    assert not failures, f"{len(failures)} programs disagree; first:\n{failures[0][1]}\ndiff {failures[0][2]} pc {failures[0][3]}"
    assert skipped <= PROGRAMS // 4, f"{skipped} of {PROGRAMS} programs did not compile"


def test_compiler_matches_c_semantics():
    """Int programs on the interpreter against CRef at N=8 and N=16, with
    early returns, while loops, && || ! ~, and variable shifts and divisors."""
    from strandedcpu.api import run_function
    rng = random.Random(7)
    failures, ran = [], 0
    for k in range(PROGRAMS * 12):
        N = 8 if k % 2 else 16
        src = Gen(rng, N, fix=False).program()
        try:
            want = CRef(src, N).run()
        except (ZeroDivisionError, IndexError, ValueError):    # undefined in C; skip
            continue
        try:
            got = run_function(src, N=N, s=64, m=128, max_steps=200000)
        except SyntaxError:                 # at N=8, a noinline call past instruction 255
            continue
        except RuntimeError as exc:         # e.g. a clobbered loop counter never halts
            got = str(exc)
        ran += 1
        if isinstance(got, str):
            failures.append((N, src, got))
            continue
        diff = {v: (got[v], want[v]) for v in ('a', 'b', 'c', 'd', 't') if got[v] != want[v]}
        if diff:
            failures.append((N, src, diff))
    assert not failures, f"{len(failures)} of {ran} differ from C; first (N={failures[0][0]}):\n{failures[0][1]}{failures[0][2]}"
    assert ran >= PROGRAMS * 10


# fix programs against exact rational arithmetic
#
# Each fix operation rounds to 2^-N, so results are compared within a
# tolerance; a program whose branch or (int) cast lands within 1/64 of a tie
# (other than an exact tie of exactly representable values) is skipped.

FIX_HELPERS = ("fix fclip(fix v) { if (v < -2.0) { return -2.0; } if (v > 2.0) { return 2.0; } return v; }\n"
               "fix fsq(fix v) { fix h; h = v * v; return h; }\n"
               "noinline fix favg(fix p, fix q) { return (p + q) / 2; }\n"
               "int fsign(fix v) { if (v < 0) { return -1; } return 1; }\n")


class FixGen:
    """fix expressions (+ - * /, unary minus, inline and noinline fix
    functions with early returns), casts, fix comparisons as values and in
    branches. Divisors are literals or fclip(e) + 3, so never near zero."""

    def __init__(self, rng):
        self.rng = rng

    def lit(self):
        return f"{self.rng.randint(-12, 12) / 4}"

    def atom(self):
        r = self.rng.random()
        if r < 0.3:
            return self.lit()
        if r < 0.85:
            return self.rng.choice('xyzw')
        return self.rng.choice(['a', 'b', '(fix) a'])

    def fexpr(self, depth=0):
        if depth > 2 or self.rng.random() < 0.3:
            return self.atom()
        op = self.rng.choice(['+', '-', '*', '/', 'neg', 'clip', 'sq', 'avg', '+', '*'])
        x, y = self.fexpr(depth + 1), self.fexpr(depth + 1)
        if op == '/':
            return f"({x} / {self.rng.choice(['2', '4', '0.5', '-2.5', f'(fclip({y}) + 3.0)'])})"
        if op == 'neg':
            return f"(-{x})"
        if op == 'clip':
            return f"fclip({x})"
        if op == 'sq':
            return f"fsq(fclip({x}))"
        if op == 'avg':
            return f"favg({x}, {y})"
        return f"({x} {op} {y})"

    def stmt(self, depth=0):
        r = self.rng.random()
        v = self.rng.choice('xyzw')
        k = self.rng.choice('ab')
        if r < 0.45:
            return f"{v} = {self.fexpr()};"
        if r < 0.55:
            return f"{v} {self.rng.choice(['+=', '-='])} {self.fexpr()};"
        if r < 0.63:
            return f"{k} = (int) {self.fexpr(1)};"
        if r < 0.70:
            return f"{k} = fsign({self.fexpr(1)});"
        if r < 0.77:
            return f"{k} = ({self.fexpr(1)} {self.rng.choice(['<', '>=', '=='])} {self.fexpr(1)}) + a;"
        if r < 0.9 and depth < 2:
            return (f"if ({self.fexpr(1)} {self.rng.choice(['<', '<=', '>', '>='])} {self.fexpr(1)}) "
                    f"{{ {self.stmt(depth + 1)} {self.stmt(depth + 1)} }} else {{ {self.stmt(depth + 1)} }}")
        if depth < 1:
            return f"for (i = 0; i < {self.rng.randint(1, 3)}; i += 1) {{ {self.stmt(1)} {self.stmt(1)} }}"
        return f"{v} = {self.fexpr()};"

    def program(self):
        init = " ".join(f"{v} = {self.lit()};" for v in 'xyzw')
        body = " ".join(self.stmt() for _ in range(self.rng.randint(3, 7)))
        return (f"int a; int b; int i; fix x; fix y; fix z; fix w;\n{FIX_HELPERS}"
                f"int main() {{ a = {self.rng.randint(-3, 3)}; b = {self.rng.randint(-3, 3)}; {init}\n"
                f"  {body}\n  return 0; }}\n")


class _FixReturn(Exception):
    pass


class Ambiguous(Exception):
    """A branch or cast too close to a tie for rounding to be ruled out."""


class FixRef:
    """Evaluates FixGen programs with fix values as exact Fractions."""

    MARGIN = Fr(1, 64)

    def __init__(self, src, N=16, exact=False):
        self.N = N
        if exact:               # numbers='rational': nothing rounds, so no tie is ambiguous
            self.MARGIN = 0
            self.exact = lambda v: True
        prog = cc.Parser(cc.lex(src)).parse_program()
        self.funcs = {f.name: f for f in prog.functions}
        self.g, self.types = {}, {}
        for d in prog.globals:
            self.types[d.name] = d.type
            self.g[d.name] = Fr(0) if d.type == 'fix' else 0
        self.budget = 10 ** 5

    def run(self):
        try:
            self.block(self.funcs['main'].body)
        except _FixReturn:
            pass
        return self.g

    def exact(self, v):
        """Representable in N fraction bits, so no rounding can have moved it."""
        return (1 << self.N) % Fr(v).denominator == 0

    def ev(self, e):
        if isinstance(e, cc.IntLiteral):
            return e.value
        if isinstance(e, cc.FixLiteral):
            return Fr(e.value)
        if isinstance(e, cc.VarRef):
            return self.g[e.name]
        if isinstance(e, cc.UnaryOp):
            v = self.ev(e.expr)
            if e.op == '-':
                return -v
            if e.op == '(fix)':
                return Fr(v)
            f = v.__floor__()       # (int) floors
            if isinstance(v, Fr) and 0 < min(v - f, f + 1 - v) < self.MARGIN:
                raise Ambiguous
            return f
        if isinstance(e, cc.BinaryOp):
            a, b = self.ev(e.left), self.ev(e.right)
            fix = isinstance(a, Fr) or isinstance(b, Fr)
            if e.op in ('<', '<=', '>', '>=', '==', '!='):
                d = abs(Fr(a) - Fr(b))
                if fix and (0 < d < self.MARGIN or (d == 0 and not self.exact(a))):
                    raise Ambiguous
                return int({'<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b, '==': a == b, '!=': a != b}[e.op])
            if e.op == '/':
                if fix:
                    return Fr(a) / Fr(b)
                q = abs(a) // abs(b)
                return q if (a < 0) == (b < 0) else -q
            return {'+': a + b, '-': a - b, '*': a * b}[e.op]
        if isinstance(e, cc.FuncCall):
            f = self.funcs[e.name]
            args = [self.ev(x) for x in e.args]
            saved = {p: (self.g.get(p), self.types.get(p)) for p in f.params}
            for p, t, v in zip(f.params, f.param_types or ['int'] * len(args), args):
                self.types[p] = t
                self.g[p] = Fr(v) if t == 'fix' else v
            try:
                self.block(f.body)
                r = 0
            except _FixReturn as ret:
                r = ret.args[0]
            for p, (v, t) in saved.items():
                if v is None:
                    self.g.pop(p, None)
                else:
                    self.g[p], self.types[p] = v, t
            return Fr(r) if f.ret_type == 'fix' else r
        raise NotImplementedError(type(e).__name__)

    def assign(self, name, v):
        self.g[name] = Fr(v) if self.types.get(name) == 'fix' else v

    def block(self, b):
        for st in b.stmts:
            self.stmt(st)

    def stmt(self, st):
        self.budget -= 1
        if self.budget < 0:
            raise RuntimeError("FixRef: step budget exhausted")
        if isinstance(st, cc.Assignment):
            v, cur = self.ev(st.value), self.g[st.target.name]
            self.assign(st.target.name, v if st.op == '=' else cur + v if st.op == '+=' else cur - v)
        elif isinstance(st, cc.VarDecl):
            self.types[st.name] = st.type
            self.assign(st.name, self.ev(st.init) if st.init else 0)
        elif isinstance(st, cc.IfStmt):
            if self.ev(st.cond):
                self.block(st.then_body)
            elif st.else_body:
                self.block(st.else_body)
        elif isinstance(st, cc.ForStmt):
            self.stmt(st.init)
            while self.ev(st.cond):
                self.block(st.body)
                self.stmt(st.update)
        elif isinstance(st, cc.ReturnStmt):
            raise _FixReturn(self.ev(st.value) if st.value else 0)
        else:
            raise NotImplementedError(type(st).__name__)


def run_strandedcpu(src, N=16, s=64, m=192, numbers='binary'):
    """Checked compile and run on the interpreter; ints and fix values
    (as Fractions) by name. A checked overflow raises ExecutionFault."""
    cfg, memory, cmds, meta = compile_c(src, s=s, m=m, n=8192, N=N, checked=True, numbers=numbers)
    vm = ISAInterpreter(cfg.s, cfg.m, cfg.N, memory, cmds, checks=meta['checks'])
    vm.run(max_steps=10 ** 7)
    out = {k: vm.memory[a] for k, a in meta['variables'].items()}
    out.update({k: (Fr(vm.memory[a], vm.memory[a + 1]) if numbers == 'rational'
                    else Fr(fix_value(vm.memory[a], vm.memory[a + 1], cfg.N)))
                for k, a in meta['fix_variables'].items()})
    return out


def test_fix_programs_match_exact_arithmetic():
    rng = random.Random(11)
    tol = Fr(1, 100)
    failures, ran = [], 0
    for _ in range(PROGRAMS * 8):
        src = FixGen(rng).program()
        try:
            want = FixRef(src).run()
            got = run_strandedcpu(src)
        except (Ambiguous, ExecutionFault):      # near a tie, or a checked overflow
            continue
        ran += 1
        diff = {v: (float(got[v]), float(want[v])) for v in 'abxyzw'
                if ((abs(got[v] - want[v]) > tol * (1 + abs(want[v]))) if isinstance(want[v], Fr) else got[v] != want[v])}
        if diff:
            failures.append((src, diff))
    assert not failures, f"{len(failures)} of {ran} differ; first:\n{failures[0][0]}{failures[0][1]}"
    assert ran >= PROGRAMS * 7


def test_rational_fix_programs_are_exact():
    """numbers='rational': the same programs must equal exact arithmetic
    exactly, ties and casts included."""
    rng = random.Random(12)
    failures, ran = [], 0
    for _ in range(PROGRAMS * 4):
        src = FixGen(rng).program()
        want = FixRef(src, exact=True).run()
        try:
            got = run_strandedcpu(src, N=32, m=256, numbers='rational')
        except ExecutionFault:           # a numerator or denominator outgrew 32 bits
            continue
        ran += 1
        diff = {v: (got[v], want[v]) for v in 'abxyzw' if got[v] != want[v]}
        if diff:
            failures.append((src, diff))
    assert not failures, f"{len(failures)} of {ran} differ; first:\n{failures[0][0]}{failures[0][1]}"
    assert ran >= PROGRAMS * 3
