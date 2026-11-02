"""Translate numeric Python functions into the compiler's C subset.

The entry is solution(), or the sole function. Supported constructs include
numeric expressions, branches, bounded-shape lists and loops; see
docs/reference/compiler.md for the language subset.

Division and decimal literals use fix values. Binary mode rounds to N
fraction bits; rational mode converts parsed float reprs to fractions. Python's
parser has already rounded float literals, so excess source digits are lost.
Integer // and % follow Python's floor/divisor-sign rules. round(x) uses
half-to-even ties, while round(x, n) uses ties away from zero. Types are
inferred per variable, so assigning a fix value promotes that variable.
Unsupported syntax raises PythonSubsetError.
"""

import ast
import copy
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union


class PythonSubsetError(SyntaxError):
    pass


_CMP = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
_ARITH = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}


class _Inline(ast.NodeTransformer):
    """Replace reads of module constants with (copies of) their values."""

    def __init__(self, consts: Dict[str, ast.AST]):
        self.consts = consts

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.consts:
            return copy.deepcopy(self.consts[node.id])
        return node


def _elem(name: str, k: int) -> str:
    return f"{name}__{k}"


class _Lists(ast.NodeTransformer):
    """Lists of constant length, one variable per element. Every assignment
    to a list name must be a list literal of the same length; reads with a
    constant index, sum/len/max/min and `for v in xs` become plain
    expressions and loops. A variable index (read or store) is lowered later
    to a compare chain. Aliasing, append and lists passed to functions are
    not supported."""

    def __init__(self, tr: '_Translator', fn: str):
        self.tr, self.fn, self.lengths = tr, fn, {}

    def run(self, f: ast.FunctionDef) -> Dict[str, int]:
        for n in ast.walk(f):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.List):
                if len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name):
                    raise PythonSubsetError("a list literal is assigned to one name")
                name, size = n.targets[0].id, len(n.value.elts)
                if self.lengths.get(name, size) != size:
                    raise PythonSubsetError(f"list '{name}' changes length")
                self.lengths[name] = size
        for n in ast.walk(f):
            for t in getattr(n, 'targets', [getattr(n, 'target', None)]):
                if (isinstance(t, ast.Name) and t.id in self.lengths and isinstance(n, (ast.Assign, ast.AugAssign))
                        and not (isinstance(n, ast.Assign) and isinstance(n.value, ast.List))):
                    raise PythonSubsetError(f"list '{t.id}' is only assigned list literals")
            if isinstance(n, ast.Name) and n.id in {_elem(k, i) for k, v in self.lengths.items() for i in range(v)}:
                raise PythonSubsetError(f"name '{n.id}' is reserved for a list element")
        for name, size in self.lengths.items():
            self.tr.extra[self.fn].update(_elem(name, k) for k in range(size))
        self.visit(f)
        indexed = {id(p.value) for p in ast.walk(f) if isinstance(p, ast.Subscript)}
        for n in ast.walk(f):
            if (isinstance(n, ast.Name) and n.id in self.lengths and isinstance(n.ctx, ast.Load)
                    and id(n) not in indexed):
                raise PythonSubsetError(f"list '{n.id}' used as a value (only xs[i], sum, len, max, min, for)")
        return self.lengths

    def elems(self, e: ast.AST) -> Optional[List[ast.AST]]:
        """The element expressions of a list literal or list name, else None."""
        if isinstance(e, ast.List):
            return e.elts
        if isinstance(e, ast.Name) and e.id in self.lengths:
            return [ast.Name(_elem(e.id, k), ast.Load()) for k in range(self.lengths[e.id])]
        return None

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        name = node.func.id if isinstance(node.func, ast.Name) else None
        xs = self.elems(node.args[0]) if len(node.args) == 1 and not node.keywords else None
        if xs is None or name not in ('sum', 'len', 'max', 'min'):
            return node
        if name == 'len':
            return ast.Constant(len(xs))
        if name == 'sum':
            total: ast.AST = ast.Constant(0)
            for x in xs:
                total = ast.BinOp(total, ast.Add(), x)
            return total
        if not xs:
            raise PythonSubsetError(f"{name}() of an empty list")
        return xs[0] if len(xs) == 1 else ast.Call(ast.Name(name, ast.Load()), xs, [])

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        xs = self.elems(node.value)
        k = node.slice
        if isinstance(k, ast.UnaryOp) and isinstance(k.op, ast.USub) and isinstance(k.operand, ast.Constant):
            k = ast.Constant(-k.operand.value)
        if xs is not None and isinstance(k, ast.Constant) and type(k.value) is int:
            if not -len(xs) <= k.value < len(xs):
                raise PythonSubsetError("list index out of range")
            i = k.value % len(xs)
            if isinstance(node.ctx, ast.Store):
                return ast.Name(xs[i].id, ast.Store()) if isinstance(xs[i], ast.Name) else node
            return xs[i]
        if isinstance(node.value, ast.List):
            raise PythonSubsetError("a list literal indexed by a variable")
        return node

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        xs = self.elems(node.iter)
        if xs is None:
            return node
        if not isinstance(node.target, ast.Name):
            raise PythonSubsetError("for over a list binds one name")
        self.tr.gen += 1
        pre = []
        name = node.iter.id if isinstance(node.iter, ast.Name) else None
        if name is not None:
            # Python iterates the list object the loop started with: element
            # stores show up in later iterations, rebinding the name does not
            rebinds = stores = False
            for n in ast.walk(ast.Module(node.body, [])):
                for t in getattr(n, 'targets', [getattr(n, 'target', None)]):
                    if isinstance(t, ast.Name) and t.id == name:
                        rebinds = True
                    if (isinstance(t, ast.Name) and t.id.startswith(f"{name}__")) or (
                            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == name):
                        stores = True
            if rebinds and stores:
                raise PythonSubsetError(f"'{name}' is both rebound and changed inside a loop over it")
            if rebinds:             # iterate a copy taken when the loop starts
                xs = [ast.Name(_elem(name, k), ast.Load()) for k in range(len(xs))]
                name = None
        if name is None:            # for v in [a, b, c]: a hidden list
            name = f"g{self.tr.gen}_list"
            self.lengths[name] = len(xs)
            self.tr.extra[self.fn].update(_elem(name, k) for k in range(len(xs)))
            pre = [ast.Assign([ast.Name(name, ast.Store())], ast.List(xs, ast.Load()))]
        k = f"g{self.tr.gen}_k"
        self.tr.extra[self.fn].add(k)
        bind = ast.Assign([node.target], ast.Subscript(ast.Name(name, ast.Load()), ast.Name(k, ast.Load()), ast.Load()))
        loop = ast.For(ast.Name(k, ast.Store()), ast.Call(ast.Name('range', ast.Load()), [ast.Constant(len(xs))], []),
                       [bind] + node.body, [])
        return pre + [loop]


class _Rename(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old, self.new = old, new

    def visit_Name(self, node: ast.Name):
        if node.id == self.old:
            return ast.Name(self.new, node.ctx)
        return node


def _decimal(v: float) -> str:
    """A float literal as C reads it: plain decimal digits, never 1e-09."""
    text = repr(v)
    if 'e' not in text and 'E' not in text:
        return text
    from decimal import Decimal
    text = format(Decimal(text), 'f')
    return text if '.' in text else text + '.0'


def _exponent(e: ast.AST) -> Optional[Fraction]:
    """Return a constant exponent as Fraction, using repr for parsed floats, or None."""
    if isinstance(e, ast.Constant) and type(e.value) in (int, float):
        return Fraction(repr(e.value)) if isinstance(e.value, float) else Fraction(e.value)
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
        v = _exponent(e.operand)
        return None if v is None else -v if isinstance(e.op, ast.USub) else v
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div):
        a, b = _exponent(e.left), _exponent(e.right)
        return None if a is None or b is None or b == 0 else a / b
    return None


def _name(fn: str, n: str) -> str:
    """Map a Python local to a name in the compiler's flat memory namespace.

    Escape underscores in fn before the __ separator so function and local
    names cannot collide. Translator-created names start with $, which Python
    identifiers cannot contain; helpers use pyh_ and expression temporaries pyt_.
    """
    f = fn.replace('_', '_1')
    return f"pyh_{f}__{n[1:]}" if n.startswith('$') else f"py_{f}__{n}"


class _Translator:
    def __init__(self, tree: ast.Module):
        self.funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
        self.math_names: Dict[str, str] = {}   # from math import floor as fl -> {'fl': 'floor'}
        consts: Dict[str, ast.AST] = {}         # RATE = 0.2 at module level
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                continue
            if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and self._constant(n.value, consts)):
                consts[n.targets[0].id] = _Inline(consts).visit(n.value)
                continue
            if isinstance(n, ast.Import) and all(a.name == 'math' and a.asname is None for a in n.names):
                continue
            if isinstance(n, ast.ImportFrom) and n.module == 'math' and n.level == 0:
                self.math_names.update({a.asname or a.name: a.name for a in n.names})
                continue
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                continue
            if self._runner(n):     # print(solution()), if __name__ == '__main__': ..., answer = solution()
                continue
            raise PythonSubsetError("only function definitions, constants and import math are allowed at top level")
        for f in self.funcs.values():           # from math import ... inside a function
            for n in ast.walk(f):
                if isinstance(n, ast.ImportFrom) and n.module == 'math' and n.level == 0:
                    self.math_names.update({a.asname or a.name: a.name for a in n.names})
        for f in self.funcs.values():           # a constant a function assigns is that function's local
            local = {t.id for n in ast.walk(f) for t in getattr(n, 'targets', [getattr(n, 'target', None)])
                     if isinstance(t, ast.Name)} | {a.arg for a in f.args.args}
            _Inline({k: v for k, v in consts.items() if k not in local}).visit(f)
        self.extra: Dict[str, set] = {fn: set() for fn in self.funcs}   # hidden locals (generator loops)
        self.gen = 0
        self.lists: Dict[str, Dict[str, int]] = {}      # function -> list name -> length
        for fn, f in self.funcs.items():
            self.lists[fn] = _Lists(self, fn).run(f)
        self.fix_vars: Dict[str, set] = {}
        self.tmp = 0
        self.temps: Dict[str, bool] = {}       # C name -> is fix
        self.free: Dict[tuple, List[str]] = {}  # (function, is fix) -> reusable temporaries
        self.taken: List[tuple] = []
        self._typing: set = set()
        self._returns: Dict[str, bool] = {}    # returns_fix, memoized once types are final
        self._final = False

    def _runner(self, n: ast.stmt) -> bool:
        """Top-level code that only runs or prints the solution, which the
        caller does instead: a call statement, an `if __name__ == ...` block,
        or an assignment from a call that no function reads."""
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            return True
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == '__name__'):
            return True
        if isinstance(n, ast.Assign) and any(isinstance(c, ast.Call) for c in ast.walk(n.value)):
            read = {x.id for f in self.funcs.values() for x in ast.walk(f)
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
            names = {t.id for t in n.targets if isinstance(t, ast.Name)}
            return len(names) == len(n.targets) and not names & read
        return False

    @staticmethod
    def _constant(e: ast.AST, consts: Dict[str, ast.AST]) -> bool:
        """Literal arithmetic over numbers and earlier constants."""
        if isinstance(e, ast.Constant):
            return type(e.value) in (int, float, bool)
        if isinstance(e, ast.Name):
            return e.id in consts
        if isinstance(e, ast.UnaryOp):
            return _Translator._constant(e.operand, consts)
        if isinstance(e, ast.BinOp):
            return _Translator._constant(e.left, consts) and _Translator._constant(e.right, consts)
        return False

    # types

    def callee(self, e: ast.Call) -> Optional[str]:
        """'abs', 'solution', 'math.floor', ...; None for anything else."""
        f = e.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'math':
            return f'math.{f.attr}'
        if isinstance(f, ast.Name):
            return f'math.{self.math_names[f.id]}' if f.id in self.math_names else f.id
        return None

    def is_fix(self, e: ast.AST, fn: str) -> bool:
        if isinstance(e, ast.Constant):
            return isinstance(e.value, float)
        if isinstance(e, ast.Name):
            return e.id in self.fix_vars[fn]
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in self.lists.get(fn, {}):
            return _elem(e.value.id, 0) in self.fix_vars[fn]
        if isinstance(e, ast.BinOp):
            if isinstance(e.op, ast.Div):
                return True
            if isinstance(e.op, (ast.FloorDiv, ast.Mod)):     # float // and % give floats
                return self.is_fix(e.left, fn) or self.is_fix(e.right, fn)
            if isinstance(e.op, ast.Pow):     # a float, fractional or negative exponent gives a float
                k = _exponent(e.right)
                return self.is_fix(e.left, fn) or (k is not None and (k.denominator != 1 or k < 0)) or any(
                    isinstance(n, ast.Constant) and isinstance(n.value, float) or isinstance(n, ast.Div)
                    for n in ast.walk(e.right))
            return self.is_fix(e.left, fn) or self.is_fix(e.right, fn)
        if isinstance(e, ast.UnaryOp):
            return isinstance(e.op, (ast.USub, ast.UAdd)) and self.is_fix(e.operand, fn)
        if isinstance(e, ast.IfExp):
            return self.is_fix(e.body, fn) or self.is_fix(e.orelse, fn)
        if isinstance(e, ast.BoolOp):
            return any(self.is_fix(v, fn) for v in e.values)
        if isinstance(e, ast.Call):
            name = self.callee(e)
            if name == 'round' and len(e.args) == 2:     # round(x, n) keeps x's type
                return self.is_fix(e.args[0], fn)
            if name in ('int', 'round', 'math.floor', 'math.ceil', 'math.isqrt'):
                return False
            if name in ('float', 'math.sqrt'):
                return True
            if name in ('abs', 'min', 'max'):
                return any(self.is_fix(a, fn) for a in e.args)
            if name == 'sum' and len(e.args) == 1 and isinstance(e.args[0], ast.GeneratorExp):
                g = e.args[0]       # the loop variable is an int from range()
                return self.is_fix(_Rename(g.generators[0].target.id, '__int').visit(copy.deepcopy(g.elt)), fn)
            if name in self.funcs:
                return self.returns_fix(name)
        return False

    def returns_fix(self, fn: str) -> bool:
        if fn in self._returns:
            return self._returns[fn]
        if fn in self._typing:
            raise PythonSubsetError(f"{fn}() is recursive (noinline functions are not reentrant)")
        self._typing.add(fn)
        try:
            fix = any(self.is_fix(n.value, fn) for n in ast.walk(self.funcs[fn])
                      if isinstance(n, ast.Return) and n.value is not None)
        finally:
            self._typing.discard(fn)
        if self._final:             # types are settled once infer() has finished
            self._returns[fn] = fix
        return fix

    def divmod_tuple(self, v: ast.AST) -> ast.AST:
        """divmod(a, b) as the tuple (a // b, a % b); anything else unchanged."""
        if isinstance(v, ast.Call) and self.callee(v) == 'divmod' and len(v.args) == 2 and not v.keywords:
            a, b = v.args
            return ast.Tuple([ast.BinOp(a, ast.FloorDiv(), b), ast.BinOp(a, ast.Mod(), b)], ast.Load())
        return v

    def infer(self):
        self._infer()
        self._final = True

    def _infer(self):
        for fn in self.funcs:
            self.fix_vars[fn] = set()
        changed = True
        while changed:
            changed = False
            for fn, f in self.funcs.items():
                for n in ast.walk(f):
                    targets = []
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.List):
                        name = n.targets[0].id
                        targets = [(ast.Name(_elem(name, k), ast.Store()), v) for k, v in enumerate(n.value.elts)]
                    elif isinstance(n, ast.Assign):
                        for t in n.targets:
                            v = self.divmod_tuple(n.value)
                            if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple):
                                targets += list(zip(t.elts, v.elts))
                            else:
                                targets.append((t, n.value))
                    elif isinstance(n, ast.AugAssign):
                        targets = [(n.target, ast.BinOp(n.target, n.op, n.value))]
                    elif isinstance(n, ast.Call) and self.callee(n) in self.funcs:
                        # a helper's parameter is fractional if any call passes it a fractional value
                        g = self.callee(n)
                        for a, v in zip(self.funcs[g].args.args, n.args):
                            if a.arg not in self.fix_vars[g] and self.is_fix(v, fn):
                                self.fix_vars[g].add(a.arg)
                                changed = True
                    for t, v in targets:
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                            t = ast.Name(_elem(t.value.id, 0), ast.Store())
                        if isinstance(t, ast.Name) and t.id not in self.fix_vars[fn] and self.is_fix(v, fn):
                            self.fix_vars[fn].add(t.id)
                            changed = True
                # a list holds one type: fractional if any element is
                for name, size in self.lists[fn].items():
                    elems = {_elem(name, k) for k in range(size)}
                    if elems & self.fix_vars[fn] and not elems <= self.fix_vars[fn]:
                        self.fix_vars[fn] |= elems
                        changed = True

    # expressions: three-address code
    # Every non-trivial subexpression goes to its own fresh variable, so no C
    # expression ever spans a call: inline and noinline calls reuse the C
    # compiler's expression temporaries. and / or keep Python's
    # short-circuit (b != 0 and a // b > 2 must not divide by zero).

    def fresh(self, fn: str, fix: bool) -> str:
        """A temporary of this function and type: reused once the statement
        that took it has finished (statement() returns it to the pool)."""
        pool = self.free.setdefault((fn, fix), [])
        if pool:
            name = pool.pop()
        else:
            self.tmp += 1
            name = f"pyt_{fn.replace('_', '_1')}__{self.tmp}"      # not py_<fn>__t<n>: a local t2 would share it
            self.temps[name] = fix
        self.taken.append((fn, fix, name))
        return name

    def value(self, e: ast.AST, fn: str, out: List[str], ind: str) -> str:
        """Operand (a name or literal) holding e; statements go to out."""
        if isinstance(e, ast.Constant) and isinstance(e.value, bool):
            return '1' if e.value else '0'
        if isinstance(e, ast.Constant) and type(e.value) in (int, float):
            return _decimal(e.value) if isinstance(e.value, float) else str(e.value)
        if isinstance(e, ast.Constant) and e.value is None:
            raise PythonSubsetError("returns None (no numeric answer)")
        if isinstance(e, ast.Name):
            if e.id in ('True', 'False'):
                return '1' if e.id == 'True' else '0'
            return _name(fn, e.id)
        fix = self.is_fix(e, fn)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.UAdd):
            return self.value(e.operand, fn, out, ind)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant) \
                and type(e.operand.value) in (int, float):
            return f"(-{self.value(e.operand, fn, out, ind)})"
        t = self.fresh(fn, fix)
        if isinstance(e, ast.UnaryOp):
            v = self.value(e.operand, fn, out, ind)
            if isinstance(e.op, ast.USub):
                out.append(f"{ind}{t} = -{v};")
            elif isinstance(e.op, ast.Not):
                out.append(f"{ind}{t} = !{v};")
            else:
                raise PythonSubsetError(f"unary {type(e.op).__name__}")
            return t
        if isinstance(e, ast.BinOp):
            a, b = self.value(e.left, fn, out, ind), self.value(e.right, fn, out, ind)
            if type(e.op) in _ARITH:
                out.append(f"{ind}{t} = {a} {_ARITH[type(e.op)]} {b};")
            elif isinstance(e.op, ast.Div):
                out.append(f"{ind}{t} = (fix) {a} / {b};")
            elif isinstance(e.op, (ast.FloorDiv, ast.Mod)) and fix:
                # Python's float // and %: floor(a / b), and a - b * floor(a / b)
                # (the sign of b), from the exact quotient; (int) on fix floors
                q, f = self.fresh(fn, True), self.fresh(fn, False)
                out.append(f"{ind}{q} = (fix) {a} / {b};")
                out.append(f"{ind}{f} = (int) {q};")
                out.append(f"{ind}{t} = (fix) {f};" if isinstance(e.op, ast.FloorDiv) else
                           f"{ind}{t} = {a} - {b} * {f};")
            elif isinstance(e.op, (ast.FloorDiv, ast.Mod)):
                f = 'py_floordiv' if isinstance(e.op, ast.FloorDiv) else 'py_mod'
                out.append(f"{ind}{t} = {f}({a}, {b});")
            elif isinstance(e.op, ast.Pow) and _exponent(e.right) is None:
                # an int exponent known only at run time: exact repeated
                # multiplication in the runtime (numbers='rational'); |n| > 4096
                # stops (step_limit), and so does an int base with n < 0 (Python
                # would give a float)
                if self.is_fix(e.right, fn):
                    raise PythonSubsetError("** needs an int exponent, or a constant one")
                out.append(f"{ind}{t} = {'fpowi' if fix else 'ipowi'}({a}, {b});")
            elif isinstance(e.op, ast.Pow):
                # a constant exponent p/q (2, 2.0, 6 / 3, 0.5, -1.5): x ** (p/q)
                # is the exact q-th root (numbers='rational'; a root that is
                # not rational stops the run: inexact_sqrt), then the integer
                # power |p|, then its reciprocal if p < 0
                k = _exponent(e.right)
                if k is None or not 1 <= k.denominator <= 64 or not -64 <= k.numerator <= 64:
                    raise PythonSubsetError("** needs a constant exponent p/q with |p| <= 64 and q <= 64")
                if k.denominator != 1:
                    r = self.fresh(fn, True)
                    root = f"fsqrt({a})" if k.denominator == 2 else f"froot({a}, {k.denominator})"
                    out.append(f"{ind}{r} = {root};")
                    a = r
                neg, k = k < 0, abs(k.numerator)
                # square and multiply; p is squared only while higher bits
                # remain, so no square exceeds the result's magnitude
                p, have = a, False
                while True:
                    if k & 1:
                        out.append(f"{ind}{t} = {t} * {p};" if have else f"{ind}{t} = {p};")
                        have = True
                    k >>= 1
                    if not k:
                        break
                    q = self.fresh(fn, fix)
                    out.append(f"{ind}{q} = {p} * {p};")
                    p = q
                if not have:
                    out.append(f"{ind}{t} = 1;")
                if neg:
                    out.append(f"{ind}{t} = (fix) 1 / {t};")
            else:
                raise PythonSubsetError(f"operator {type(e.op).__name__}")
            return t
        if isinstance(e, ast.Compare):
            left = self.value(e.left, fn, out, ind)
            body, inner = out, ind
            closes = []
            for k, (op, right) in enumerate(zip(e.ops, e.comparators)):
                if type(op) not in _CMP:
                    raise PythonSubsetError(f"comparison {type(op).__name__}")
                r = self.value(right, fn, body, inner)
                body.append(f"{inner}{t} = {left} {_CMP[type(op)]} {r};")
                if k < len(e.ops) - 1:              # a < b < c: c only if a < b
                    body.append(f"{inner}if ({t}) {{")
                    closes.append(inner)
                    inner += "    "
                left = r
            for c in reversed(closes):
                body.append(f"{c}}}")
            return t
        if isinstance(e, ast.BoolOp):
            # Python's and/or yield an operand, not a bool: x and y is x if x
            # is falsy else y; x or y is x if x is truthy else y.
            out.append(f"{ind}{t} = {self.value(e.values[0], fn, out, ind)};")
            for v in e.values[1:]:
                out.append(f"{ind}if ({t if isinstance(e.op, ast.And) else '!' + t}) {{")
                out.append(f"{ind}    {t} = {self.value(v, fn, out, ind + '    ')};")
                out.append(f"{ind}}}")
            return t
        if isinstance(e, ast.IfExp):
            c = self.value(e.test, fn, out, ind)
            out.append(f"{ind}if ({c}) {{")
            out.append(f"{ind}    {t} = {self.value(e.body, fn, out, ind + '    ')};")
            out.append(f"{ind}}} else {{")
            out.append(f"{ind}    {t} = {self.value(e.orelse, fn, out, ind + '    ')};")
            out.append(f"{ind}}}")
            return t
        if isinstance(e, ast.Subscript):
            j = self.list_index(e, fn, out, ind)
            name = e.value.id
            for k in range(self.lists[fn][name]):
                out.append(f"{ind}if ({j} == {k}) {{ {t} = {_name(fn, _elem(name, k))}; }}")
            return t
        if (isinstance(e, ast.Call) and self.callee(e) == 'sum' and len(e.args) == 1 and not e.keywords
                and isinstance(e.args[0], (ast.GeneratorExp, ast.Call))):
            return self.sum_loop(e.args[0], t, fix, fn, out, ind)
        if isinstance(e, ast.Call) and self.callee(e) is not None and not e.keywords:
            name = self.callee(e)
            args = [self.value(a, fn, out, ind) for a in e.args]
            arg_fix = len(args) == 1 and self.is_fix(e.args[0], fn)
            if name == 'abs' and len(args) == 1:
                out.append(f"{ind}{t} = {'py_abs' if fix else 'abs'}({args[0]});")
            elif name == 'math.sqrt' and len(args) == 1:      # exact, or the run stops (inexact_sqrt)
                out.append(f"{ind}{t} = fsqrt({args[0]});")
            elif name == 'math.isqrt' and len(args) == 1:
                if arg_fix:
                    raise PythonSubsetError("math.isqrt() takes an int")
                out.append(f"{ind}{t} = isqrt({args[0]});")
            elif name in ('min', 'max') and len(args) >= 2:
                out.append(f"{ind}{t} = {args[0]};")
                for v in args[1:]:
                    if fix:     # the MAX builtin is int-only; fix compares exactly
                        out.append(f"{ind}if ({v} {'>' if name == 'max' else '<'} {t}) {{ {t} = {v}; }}")
                    else:
                        out.append(f"{ind}{t} = {name}({t}, {v});")
            elif name == 'round' and len(args) == 2:
                n = e.args[1].value if isinstance(e.args[1], ast.Constant) else None
                if type(n) is not int or not 0 <= n <= 6:
                    raise PythonSubsetError("round(x, n) needs a constant n in 0..6")
                if not fix:                                 # round(int, n) is the int
                    out.append(f"{ind}{t} = {args[0]};")
                else:
                    # round x * 10^n by the school rule, then back: a tie goes
                    # away from zero (1157.625 -> 1157.63, -1.005 -> -1.01;
                    # decimal.ROUND_HALF_UP), not Python's half to even
                    sc, r, d = self.fresh(fn, True), self.fresh(fn, False), self.fresh(fn, True)
                    out.append(f"{ind}{sc} = {args[0]} * {10 ** n};")
                    out.append(f"{ind}{r} = (int) {sc};")
                    out.append(f"{ind}{d} = {sc} - {r};")
                    out.append(f"{ind}if ({d} > 0.5) {{ {r} += 1; }} else {{ if ({d} == 0.5) {{ if ({sc} > 0) {{ {r} += 1; }} }} }}")
                    out.append(f"{ind}{t} = (fix) {r} / {10 ** n};")
            elif name == 'float' and len(args) == 1:
                out.append(f"{ind}{t} = (fix) {args[0]};")
            elif name in ('math.floor', 'math.ceil', 'round') and len(args) == 1:
                out.append(f"{ind}{t} = (int) {args[0]};")       # (int) on fix floors
                if arg_fix and name == 'math.ceil':
                    out.append(f"{ind}if ({t} != {args[0]}) {{ {t} += 1; }}")
                elif arg_fix and name == 'round':
                    # Python rounds half to even: 2.5 -> 2, 3.5 -> 4, -2.5 -> -2
                    d = self.fresh(fn, True)
                    out.append(f"{ind}{d} = {args[0]} - {t};")
                    out.append(f"{ind}if ({d} > 0.5) {{ {t} += 1; }} else {{ if ({d} == 0.5) {{ {t} += {t} & 1; }} }}")
            elif name == 'int' and len(args) == 1:
                src_fix = self.is_fix(e.args[0], fn)
                if src_fix:
                    # Python int() truncates toward zero; (int) on fix floors
                    out.append(f"{ind}{t} = (int) {args[0]};")
                    out.append(f"{ind}if ({args[0]} < 0) {{ if ({t} != {args[0]}) {{ {t} += 1; }} }}")
                else:
                    out.append(f"{ind}{t} = {args[0]};")
            elif name in self.funcs:
                if len(args) != len(self.funcs[name].args.args):
                    raise PythonSubsetError(f"{name}() takes {len(self.funcs[name].args.args)} arguments")
                out.append(f"{ind}{t} = pyf_{name}({', '.join(args)});")
            else:
                raise PythonSubsetError(f"call to {name}()")
            return t
        raise PythonSubsetError(f"unsupported expression: {type(e).__name__}")

    def list_index(self, e: ast.Subscript, fn: str, out: List[str], ind: str) -> str:
        """Int temp holding the checked, non-negative index of xs[i]. Out of
        range reads a one-word array past its end: a runtime_error, where
        Python raises IndexError."""
        if not (isinstance(e.value, ast.Name) and e.value.id in self.lists.get(fn, {})):
            raise PythonSubsetError("indexing works on list names only")
        if self.is_fix(e.slice, fn):
            raise PythonSubsetError("list index must be an int")
        size = self.lists[fn][e.value.id]
        j = self.fresh(fn, False)
        out.append(f"{ind}{j} = {self.value(e.slice, fn, out, ind)};")
        out.append(f"{ind}if ({j} < 0) {{ {j} += {size}; }}")
        out.append(f"{ind}if ({j} < 0 || {j} >= {size}) {{ py_oobi = 1; py_oobv = py_oob[py_oobi]; }}")
        return j

    def sum_loop(self, it: ast.AST, t: str, fix: bool, fn: str, out: List[str], ind: str) -> str:
        """sum(range(...)) or sum(expr for v in range(...) if cond) as a loop
        adding into t. The loop variable gets a hidden name, so it does not
        touch a local of the same name (Python scopes it to the generator)."""
        if isinstance(it, ast.Call):
            gen = ast.GeneratorExp(ast.Name('__v', ast.Load()), [ast.comprehension(ast.Name('__v', ast.Store()), it, [], 0)])
        else:
            gen = it
        if len(gen.generators) != 1 or gen.generators[0].is_async:
            raise PythonSubsetError("sum() over one for clause")
        g = gen.generators[0]
        if not isinstance(g.target, ast.Name):
            raise PythonSubsetError("sum(): loop variable must be a name")
        self.gen += 1
        hidden = f"$g{self.gen}_{g.target.id}"
        self.extra[fn].add(hidden)
        rename = _Rename(g.target.id, hidden)
        elt = rename.visit(copy.deepcopy(gen.elt))
        ifs = [rename.visit(copy.deepcopy(c)) for c in g.ifs]
        acc = f"$g{self.gen}_acc"
        self.extra[fn].add(acc)
        if fix:
            self.fix_vars[fn].add(acc)
        body: List[ast.stmt] = [ast.AugAssign(ast.Name(acc, ast.Store()), ast.Add(), elt)]
        for c in reversed(ifs):
            body = [ast.If(c, body, [])]
        loop = ast.For(ast.Name(hidden, ast.Store()), g.iter, body, [])
        out.append(f"{ind}{_name(fn, acc)} = 0;")
        out.extend(self.stmt(loop, fn, ind))
        out.append(f"{ind}{t} = {_name(fn, acc)};")
        return t

    # statements

    def block(self, body: List[ast.stmt], fn: str, indent: str) -> List[str]:
        out = []
        for s in body:
            out.extend(self.stmt(s, fn, indent))
        return out

    def stmt(self, s: ast.stmt, fn: str, ind: str) -> List[str]:
        """Translate one statement; its temporaries (including a loop's flag
        and bound, live for the whole loop) are free again afterwards."""
        mark = len(self.taken)
        try:
            return self._stmt(s, fn, ind)
        finally:
            for f, fix, name in self.taken[mark:]:
                self.free.setdefault((f, fix), []).append(name)
            del self.taken[mark:]

    def _stmt(self, s: ast.stmt, fn: str, ind: str) -> List[str]:
        out: List[str] = []
        if isinstance(s, ast.Assign) and isinstance(s.value, ast.List):
            name = s.targets[0].id
            vals = []
            for k, e in enumerate(s.value.elts):      # every value before any element changes
                v = self.value(e, fn, out, ind)
                c = self.fresh(fn, self.is_fix(ast.Name(_elem(name, k), ast.Load()), fn))
                out.append(f"{ind}{c} = {v};")
                vals.append(c)
            return out + [f"{ind}{_name(fn, _elem(name, k))} = {v};" for k, v in enumerate(vals)]
        if (isinstance(s, (ast.Assign, ast.AugAssign))
                and isinstance(s.targets[0] if isinstance(s, ast.Assign) else s.target, ast.Subscript)):
            t = s.targets[0] if isinstance(s, ast.Assign) else s.target
            if isinstance(s, ast.Assign) and len(s.targets) != 1:
                raise PythonSubsetError("one target for an indexed store")
            rhs = s.value if isinstance(s, ast.Assign) else ast.BinOp(ast.Subscript(t.value, t.slice, ast.Load()), s.op, s.value)
            v = self.value(rhs, fn, out, ind)
            j = self.list_index(t, fn, out, ind)
            return out + [f"{ind}if ({j} == {k}) {{ {_name(fn, _elem(t.value.id, k))} = {v}; }}"
                          for k in range(self.lists[fn][t.value.id])]
        if isinstance(s, ast.Assign):
            if all(isinstance(t, ast.Name) for t in s.targets):
                v = self.value(s.value, fn, out, ind)
                return out + [f"{ind}{_name(fn, t.id)} = {v};" for t in s.targets]
            t, value = s.targets[0], self.divmod_tuple(s.value)
            if (len(s.targets) != 1 or not isinstance(t, ast.Tuple) or not isinstance(value, ast.Tuple)
                    or len(t.elts) != len(value.elts) or not all(isinstance(e, ast.Name) for e in t.elts)):
                raise PythonSubsetError("assignment to names, or a, b = x, y")
            # a, b = b, a % b: every value first, then the names. A value that
            # is itself one of the names is copied before any name changes.
            names = {_name(fn, e.id) for e in t.elts}
            vals = []
            for e in value.elts:
                v = self.value(e, fn, out, ind)
                if v in names:
                    c = self.fresh(fn, self.is_fix(e, fn))
                    out.append(f"{ind}{c} = {v};")
                    v = c
                vals.append(v)
            return out + [f"{ind}{_name(fn, e.id)} = {v};" for e, v in zip(t.elts, vals)]
        if isinstance(s, ast.AugAssign):
            if not isinstance(s.target, ast.Name):
                raise PythonSubsetError("augmented assignment to a name only")
            v = self.value(ast.BinOp(s.target, s.op, s.value), fn, out, ind)
            return out + [f"{ind}{_name(fn, s.target.id)} = {v};"]
        if isinstance(s, ast.If):
            c = self.value(s.test, fn, out, ind)
            out += [f"{ind}if ({c}) {{"] + self.block(s.body, fn, ind + "    ")
            if s.orelse:
                out += [f"{ind}}} else {{"] + self.block(s.orelse, fn, ind + "    ")
            return out + [f"{ind}}}"]
        if isinstance(s, ast.While):
            if s.orelse:
                raise PythonSubsetError("while ... else")
            # while (1) costs nothing, and the test at the top of the body
            # runs again after a continue
            test: List[str] = []
            if not (isinstance(s.test, ast.Constant) and s.test.value):     # while True: no test
                c = self.value(s.test, fn, test, ind + "    ")
                test.append(f"{ind}    if (!{c}) {{ break; }}")
            return (out + [f"{ind}while (1) {{"] + test
                    + self.block(s.body, fn, ind + "    ") + [f"{ind}}}"])
        if isinstance(s, ast.For):
            it = s.iter
            if (s.orelse or not isinstance(s.target, ast.Name) or not isinstance(it, ast.Call)
                    or not isinstance(it.func, ast.Name) or it.func.id != 'range' or not 1 <= len(it.args) <= 3):
                raise PythonSubsetError("for loops must be: for name in range(...)")
            step = 1
            if len(it.args) == 3:
                st = it.args[2]
                if isinstance(st, ast.UnaryOp) and isinstance(st.op, ast.USub) and isinstance(st.operand, ast.Constant):
                    step = -st.operand.value
                elif isinstance(st, ast.Constant) and isinstance(st.value, int):
                    step = st.value
                else:
                    raise PythonSubsetError("range() step must be a constant")
                if step == 0:
                    raise PythonSubsetError("range() step is zero")
            bounds = [self.value(a, fn, out, ind) for a in it.args[:2]]
            start, stop = ('0', bounds[0]) if len(bounds) == 1 else (bounds[0], bounds[1])
            # A hidden counter drives the loop and is copied into the target
            # each iteration: assigning the target in the body (or reusing it
            # in an inner loop) does not change the iteration, as in Python.
            end = self.fresh(fn, False)
            k = self.fresh(fn, False)
            v = _name(fn, s.target.id)
            cmp = '<' if step > 0 else '>'
            inc = f"{k} += {step}" if step > 0 else f"{k} -= {-step}"
            return (out + [f"{ind}{end} = {stop};", f"{ind}for ({k} = {start}; {k} {cmp} {end}; {inc}) {{",
                           f"{ind}    {v} = {k};"]
                    + self.block(s.body, fn, ind + "    ") + [f"{ind}}}"])
        if isinstance(s, ast.Return):
            v = self.value(s.value, fn, out, ind) if s.value is not None else '0'
            return out + [f"{ind}return {v};"]
        if isinstance(s, ast.Pass):
            return []
        if isinstance(s, ast.Import) and all(a.name == 'math' for a in s.names):
            return []                                   # import math inside a function
        if isinstance(s, ast.ImportFrom) and s.module == 'math':
            return []                                   # its names were registered up front
        if isinstance(s, (ast.Break, ast.Continue)):
            return [f"{ind}{'break' if isinstance(s, ast.Break) else 'continue'};"]
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            return []
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and self.callee(s.value) == 'print':
            return []                                   # output is not the result
        raise PythonSubsetError(f"unsupported statement: {type(s).__name__}")

    def translate(self, entry: Optional[str]) -> Tuple[str, str, List[str], bool]:
        self.infer()
        if entry is None:
            entry = 'solution' if 'solution' in self.funcs else next(iter(self.funcs)) if len(self.funcs) == 1 else None
        if entry not in self.funcs:
            raise PythonSubsetError("no entry function (name it solution)")
        decls, bodies = {}, []
        for fn, f in self.funcs.items():
            names = {n.id for n in ast.walk(f) if isinstance(n, ast.Name) and n.id not in self.funcs
                     and n.id not in ('range', 'abs', 'min', 'max', 'int', 'True', 'False')}
            for n in names | {a.arg for a in f.args.args}:
                if not n.isidentifier():
                    continue
                decls[_name(fn, n)] = n in self.fix_vars[fn]
        params = [a.arg for a in self.funcs[entry].args.args]
        if 'result' in params:
            raise PythonSubsetError("a parameter may not be named result")
        for fn, f in self.funcs.items():
            if f.args.vararg or f.args.kwarg or f.args.kwonlyargs or f.args.defaults:
                raise PythonSubsetError("plain positional parameters only")
            ret = 'fix' if self.returns_fix(fn) else 'int'
            ps = ', '.join(f"{'fix' if a.arg in self.fix_vars[fn] else 'int'} {_name(fn, a.arg)}_in" for a in f.args.args)
            body = self.block(f.body, fn, "    ")
            copies = [f"    {_name(fn, a.arg)} = {_name(fn, a.arg)}_in;" for a in f.args.args]
            bodies.append(f"noinline {ret} pyf_{fn}({ps}) {{\n" + "\n".join(copies + body) + "\n    return 0;\n}")
            for n in self.extra[fn]:            # hidden locals made while translating the body
                decls[_name(fn, n)] = n in self.fix_vars[fn]
        result_fix = self.returns_fix(entry)
        lines = [f"{'fix' if fx else 'int'} {n};" for n, fx in sorted(decls.items())]
        lines += [f"{'fix' if fx else 'int'} {n};" for n, fx in self.temps.items()]
        lines += [f"{'fix' if result_fix else 'int'} result;"]
        lines += [f"{'fix' if p in self.fix_vars[entry] else 'int'} {p};" for p in params]
        lines = ["int py_oob[1]; int py_oobi; int py_oobv;"] + lines     # first: indexed arrays live low
        lines += ["int py_q; int py_r;",
                  # floor = truncation, one lower when inexact and the signs differ;
                  # |q*b| <= |a|, so the product cannot overflow
                  "int py_floordiv(int a, int b) { py_q = a / b; py_r = py_q * b;"
                  " if (py_r != a) { if ((a < 0) != (b < 0)) { py_q -= 1; } } return py_q; }",
                  "int py_mod(int a, int b) { py_r = a % b;"
                  " if (py_r != 0) { if ((py_r < 0) != (b < 0)) { py_r += b; } } return py_r; }",
                  "fix py_abs(fix a) { return a * (1 - 2 * (a < 0)); }"]
        lines += bodies
        lines.append(f"int main() {{ result = pyf_{entry}({', '.join(params)}); return 0; }}")
        return "\n".join(lines) + "\n", entry, tuple(params), result_fix


@lru_cache(maxsize=4096)
def python_to_c(src: str, entry: Optional[str] = None):
    """C source for the Python subset; returns (c_source, entry, params,
    result_is_fix). The result lands in the global `result`; parameters
    become globals of the same names. Cached: the translation is pure."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise PythonSubsetError(f"Python syntax: {exc}") from None
    return _Translator(tree).translate(entry)


def score_python(src: str, inputs: Dict[str, Union[int, float]], option_texts: Dict[str, str], *,
                 N: int, entry: Optional[str] = None, m: Optional[int] = None, **kw) -> Dict[str, object]:
    """Score a Python-subset program and include its translated c_source.

    Unsupported syntax abstains as compile_error. Keyword arguments pass through
    to score_options. numbers="rational" uses fraction arithmetic after float
    literal parsing; it cannot recover source digits lost by the parser.
    The default binary fix representation has N fraction bits. Integer division
    remainders and fractional int() casts appear in inexact_ops.
    """
    return score_python_batch([{'src': src, 'inputs': inputs, 'option_texts': option_texts}],
                              N=N, entry=entry, m=m, **kw)[0]


def score_python_batch(items: List[Dict[str, object]], *, N: int, entry: Optional[str] = None,
                       m: Optional[int] = None, **kw) -> List[Dict[str, object]]:
    """Score items containing src, inputs, option_texts and optional entry.

    CUDA execution and compile_workers batching follow score_options_batch.
    """
    import hashlib
    from strandedcpu.api import score_options_batch, parse_option_number
    if m is None:           # memory words; the rational runtime needs room of its own
        m = 512 if kw.get('numbers') == 'rational' else 256
    results: List[Optional[Dict[str, object]]] = [None] * len(items)
    workers = kw.pop('compile_workers', 0)
    translated: List[Optional[str]] = [None] * len(items)
    if workers and workers > 1 and len(items) > 1:
        from strandedcpu.api import precompile
        translated = precompile([('python', it['src'], it.get('entry', entry), kw.get('s', 64), m, N,
                                  kw.get('numbers', 'binary')) for it in items], workers,
                                pack=kw.get('backend') == 'cuda')
    calls, where, sources = [], [], []
    for k, item in enumerate(items):
        src, options = item['src'], item['option_texts']
        try:
            c_src = translated[k] or python_to_c(src, item.get('entry', entry))[0]
        except PythonSubsetError as exc:
            parsed = {key: parse_option_number(t) for key, t in options.items()}
            results[k] = {'status': 'abstain', 'abstain_reason': 'compile_error', 'fault_detail': str(exc),
                          'value': None, 'value_raw': None,
                          'parsed': {key: (None if v is None else int(v) if isinstance(v, int) else float(v))
                                     for key, v in parsed.items()},
                          'match': {}, 'match_kind': None, 'inexact_ops': {'int_div_remainder': 0, 'fix_rounding': 0},
                          'steps': 0, 'latency_s': 0.0, 'backend': kw.get('backend', 'interpreter'), 'N': N,
                          'program_sha256': hashlib.sha256(src.encode()).hexdigest(), 'c_source': None}
            continue
        calls.append({'program': c_src, 'inputs': item.get('inputs') or {}, 'option_texts': options})
        where.append(k)
        sources.append((src, c_src))
    for k, (src, c_src), r in zip(where, sources, score_options_batch(calls, N=N, result='result', m=m, **kw)):
        r['program_sha256'] = hashlib.sha256(src.encode()).hexdigest()
        r['c_source'] = c_src
        results[k] = r
    return results
