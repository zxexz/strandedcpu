"""Compare the Python subset compiler with CPython.

Random programs check values and overflow detection on the checked interpreter.
Targeted cases cover rational arithmetic, rounding, lists, and helper calls.
"""

import ast
import random

from strandedcpu.python import PythonSubsetError, python_to_c, score_python, score_python_batch

N = 32
LIM = 1 << (N - 1)


class Gen:
    def __init__(self, rng, lists=False):
        self.rng, self.lists = rng, lists

    def atom(self):
        if self.lists and self.rng.random() < 0.25:     # xs is a 3-element list of ints
            k = self.rng.random()
            if k < 0.4:
                return f"xs[{self.rng.randint(-3, 2)}]"
            if k < 0.7:
                return f"xs[{self.rng.choice(['a', 'b', 'c'])} % 3]"
            return f"{self.rng.choice(['sum', 'max', 'min', 'len'])}(xs)"
        return self.rng.choice(['a', 'b', 'c', str(self.rng.randint(-50, 50))])

    def expr(self, d=0):
        if d > 2 or self.rng.random() < 0.3:
            return self.atom()
        k = self.rng.random()
        x, y = self.expr(d + 1), self.expr(d + 1)
        if k < 0.4:
            return f"({x} {self.rng.choice(['+', '-', '*'])} {y})"
        if k < 0.55:
            return f"({x} {self.rng.choice(['//', '%'])} {self.rng.choice(['3', '-7', '5', '(abs(' + y + ') + 1)'])})"
        if k < 0.7:
            return f"({x} {self.rng.choice(['<', '<=', '>', '>=', '==', '!='])} {y})"
        if k < 0.8:
            return f"({x} {self.rng.choice(['and', 'or'])} {y})"
        if k < 0.85:
            f = self.rng.choice(['abs', 'min', 'max', 'max3'])
            if f == 'max3':
                return f"{self.rng.choice(['min', 'max'])}({x}, {y}, {self.expr(d + 1)})"
            return f"abs({x})" if f == 'abs' else f"{f}({x}, {y})"
        if k < 0.9:
            f = self.rng.choice(['round', 'math.floor', 'math.ceil', 'pow'])
            if f == 'pow':
                return f"({x} ** {self.rng.randint(0, 2)})"
            return f"{f}({x} / {self.rng.choice(['2', '3', '4', '-7'])})"
        return f"({x} if {self.expr(d + 1)} else {y})"

    def body(self, ind="    ", d=0, loop=False):
        lines = []
        for _ in range(self.rng.randint(2, 5)):
            r = self.rng.random()
            v = self.rng.choice(['a', 'b', 'c'])
            if loop and r < 0.08:
                lines.append(f"{ind}if {self.expr(1)}:")
                lines.append(f"{ind}    {self.rng.choice(['break', 'continue'])}")
            elif self.lists and r < 0.2:
                k = self.rng.random()
                if k < 0.35:
                    lines.append(f"{ind}xs[({self.expr(1)}) % 3] {self.rng.choice(['=', '+=', '-='])} {self.expr(1)}")
                elif k < 0.7:
                    lines.append(f"{ind}xs = [{self.expr(1)}, {self.expr(1)}, xs[0]]")
                elif d < 2:
                    lines.append(f"{ind}for v{d} in xs:")
                    lines.append(f"{ind}    {v} += v{d}")
                    lines += self.body(ind + "    ", d + 1, loop=True)
                else:
                    lines.append(f"{ind}xs[{self.rng.randint(-3, 2)}] = {self.expr(1)}")
            elif r < 0.1:     # simultaneous: every value before any name changes
                x, y = self.rng.sample(['a', 'b', 'c'], 2)
                lines.append(f"{ind}{x}, {y} = {self.expr(1)}, {self.expr(1)}")
            elif r < 0.5 or d > 1:
                lines.append(f"{ind}{v} {self.rng.choice(['=', '+=', '-='])} {self.expr()}")
            elif r < 0.75:
                lines.append(f"{ind}if {self.expr(1)}:")
                lines += self.body(ind + "    ", d + 1)
                lines.append(f"{ind}else:")
                lines += self.body(ind + "    ", d + 1)
            elif r < 0.88:
                lines.append(f"{ind}for i in range({self.rng.randint(0, 4)}):")
                lines += self.body(ind + "    ", d + 1, loop=True)
            else:               # its own counter per depth, bumped before any continue
                lines += [f"{ind}j{d} = 0", f"{ind}while j{d} < {self.rng.randint(0, 4)}:", f"{ind}    j{d} += 1"]
                lines += self.body(ind + "    ", d + 1, loop=True)
        return lines

    def program(self):
        head = "    xs = [a, b, 7]\n" if self.lists else ""
        return ("import math\n\ndef solution(a, b, c):\n" + head + "\n".join(self.body())
                + f"\n    return {self.expr()}\n")


def python_value(src, inputs):
    env = {}
    exec(src, env)
    return int(env['solution'](**inputs))


def test_random_programs_match_cpython():
    rng = random.Random(7)
    checked = overflow = 0
    for _ in range(300):
        src = Gen(rng).program()
        inputs = {k: rng.randint(-30, 30) for k in 'abc'}
        try:
            want = python_value(src, inputs)
        except OverflowError:       # a huge int (from **) met float division
            overflow += 1
            continue
        r = score_python(src, inputs, {'x': str(want)}, N=N)
        if r['abstain_reason'] == 'overflow':
            overflow += 1
            continue
        assert r['status'] == 'ok' and r['value'] == want, (src, inputs, want, r['value'], r['abstain_reason'], r['fault_detail'])
        assert -LIM <= want < LIM
        checked += 1
    assert checked > 250


def test_semantics_that_differ_from_c():
    cases = [
        ("def solution(a, b):\n    return (a // b) * 1000 + a % b\n", [(a, b) for a in range(-9, 10) for b in (-4, -1, 3)]),
        ("def solution(a, b):\n    return int(a / b) if b else 99\n", [(a, b) for a in (-7, 7, 8) for b in (-2, 0, 3)]),
        ("def solution(a, b):\n    return 1 if (b != 0 and a // b > 2) else 0\n", [(9, 0), (9, 2), (-9, 2)]),
        ("def solution(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n", [(1071, 462), (-12, 18), (7, 0)]),
        ("def solution(a, b):\n    x = y = a\n    x, y = y + b, x - b\n    return x * 100 + y\n", [(3, 1), (-5, 2)]),
    ]
    for src, args in cases:
        for a, b in args:
            want = python_value(src, {'a': a, 'b': b})
            r = score_python(src, {'a': a, 'b': b}, {'x': str(want)}, N=16)
            assert r['status'] == 'ok' and r['value'] == want, (src, a, b, r['value'])


def test_unsupported_constructs_abstain():
    for src in ("x = 1\n", "def solution(a):\n    return [a]\n", "def solution(a):\n    return a ** a\n",
                "def solution(a):\n    while a:\n        break\n    else:\n        a = 2\n    return a\n",
                "def solution(a):\n    return f(a)\n", "def solution(a):\n    return solution(a)\n",
                "def solution(a):\n    return round(a / 3, a)\n", "import os\n\ndef solution(a):\n    return a\n"):
        r = score_python(src, {'a': 1}, {'x': '1'}, N=16)
        assert r['status'] == 'abstain' and r['abstain_reason'] == 'compile_error', src


def test_early_return_and_helpers():
    src = """def helper(x):
    t = 0
    for i in range(x):
        t += i
    return t

def solution(n):
    for k in range(n, 0, -1):
        if helper(k) < 20:
            return k
    return -1
"""
    r = score_python(src, {'n': 10}, {'x': '6', 'y': '5'}, N=16)
    assert r['status'] == 'ok' and r['value'] == 6


def test_common_idioms_match_cpython():
    """Compare imports, builtins, and control flow with CPython."""
    cases = [
        ("import math\n\ndef solution(a):\n    return math.ceil(a / 4) * 100 + math.floor(-a / 4)\n", {'a': 10}),
        ("from math import ceil as up\n\ndef solution(a):\n    return up(a / 3)\n", {'a': 7}),
        ("def solution(a):\n    return a ** 0 + a ** 1 + a ** 2 + a ** 5\n", {'a': -3}),
        ("def solution(a):\n    return round(a / 4) * 100 + round(-a / 4) * 10 + round(7 / 2)\n", {'a': 10}),
        ("def solution(a):\n    q, r = divmod(a, 7)\n    return q * 10 + r\n", {'a': -30}),
        ("def solution(a, b):\n    return max(a, b, 3) * 10 + min(a, b, 3)\n", {'a': 1, 'b': 5}),
        ("def solution(a, b):\n    return max(a / 2, b / 4, 1.25)\n", {'a': 1, 'b': 6}),
        ("def solution(a):\n    return float(a) / 2\n", {'a': 9}),
        ("def solution(n):\n    i = 0\n    while True:\n        i += 1\n        if i * i > n:\n            break\n    return i\n", {'n': 50}),
        ("def solution(n):\n    s = 0\n    i = 0\n    while i < n:\n        i += 1\n        if i % 3 == 0:\n            continue\n        s += i\n    return s\n", {'n': 10}),
        ("def solution(n):\n    s = 0\n    for i in range(n):\n        if i % 2:\n            continue\n        if i > 6:\n            break\n        s += i\n    return s\n", {'n': 20}),
    ]
    for src, inputs in cases:
        env = {}
        exec(src, env)
        want = env['solution'](**inputs)
        r = score_python(src, inputs, {'x': str(want)}, N=32)
        assert r['status'] == 'ok' and r['value'] == want, (src, want, r['value'], r['abstain_reason'], r['fault_detail'])


def test_rounding_slack_is_opt_in():
    src = "def solution(price):\n    return price * 0.8\n"          # 0.8 is not a binary fraction
    opts = {'A': '40', 'B': '41', 'C': '44'}
    exact = score_python(src, {'price': 50}, opts, N=32)
    assert exact['abstain_reason'] == 'no_match' and exact['value'] != 40
    slack = score_python(src, {'price': 50}, opts, N=32, rounding_slack=1e-6)
    assert slack['status'] == 'ok' and slack['match']['A'] and slack['match_kind'] == 'within_slack'
    money = score_python("def solution(a, b):\n    return round(a / b, 2)\n", {'a': 1, 'b': 8}, {'A': '0.12', 'B': '0.13'},
                         N=32, rounding_slack=1e-6)           # ties away from zero: 0.125 -> 0.13 (CPython: 0.12)
    assert money['status'] == 'ok' and money['match']['B']
    near = score_python(src, {'price': 50}, {'A': '40', 'B': '40.0000001'}, N=32, rounding_slack=1e-6)
    assert near['abstain_reason'] == 'multi_match'
    # no rounding happened: the slack does not apply
    whole = score_python("def solution(a):\n    return a * 0.5\n", {'a': 9}, {'A': '4.5', 'B': '4.5000001'},
                         N=32, rounding_slack=1e-6)
    assert whole['status'] == 'ok' and whole['match_kind'] == 'exact' and whole['match'] == {'A': True, 'B': False}


def test_rational_mode():
    """numbers='rational': int-valued programs still agree with CPython, and
    decimal arithmetic is exact where binary fix only comes close."""
    rng = random.Random(8)
    checked = 0
    for _ in range(150):
        src = Gen(rng).program()
        inputs = {k: rng.randint(-30, 30) for k in 'abc'}
        try:
            want = python_value(src, inputs)
        except OverflowError:
            continue
        r = score_python(src, inputs, {'x': str(want)}, N=N, numbers='rational')
        if r['abstain_reason'] == 'overflow':
            continue
        assert r['status'] == 'ok' and r['value'] == want, (src, inputs, want, r['value'], r['abstain_reason'])
        checked += 1
    assert checked > 100
    cases = [("def solution(price):\n    return price * 0.8\n", {'price': 50}, '40'),
             ("def solution(cost):\n    return round(cost * 1.07, 2)\n", {'cost': 40}, '42.8'),
             ("def solution(p, r):\n    return p * (1 + r / 100) ** 2\n", {'p': 1000, 'r': 5}, '1102.5'),
             ("def solution(a):\n    return a * 0.1 + a * 0.2 == a * 0.3\n", {'a': 7}, '1'),
             ("def solution(x):\n    return int(x * 0.15 + 0.5)\n", {'x': 10}, '2')]
    for src, inputs, gold in cases:
        r = score_python(src, inputs, {'gold': gold, 'other': '-1'}, N=32, numbers='rational')
        assert r['status'] == 'ok' and r['match']['gold'] and r['match_kind'] == 'exact', (src, r['value'])


def test_lists_constants_and_generators():
    """Constant-length lists (one variable per element), module constants,
    print(), sum() over range and generators, in both number modes."""
    cases = [
        ("RATE = 3\nBASE = RATE * 10\n\ndef solution(p):\n    return p * RATE + BASE\n", {'p': 7}),
        ("N = 5\n\ndef solution(p):\n    N = p + 1\n    return N * 2\n", {'p': 4}),
        ("def solution(x):\n    print('x is', x)\n    return x * 2\n", {'x': 21}),
        ("def solution(n):\n    return sum(range(1, n + 1)) + sum(i * i for i in range(n) if i % 2)\n", {'n': 9}),
        ("def solution(n):\n    i = 100\n    s = sum(i for i in range(n))\n    return s + i\n", {'n': 7}),
        ("def solution(a):\n    prices = [a, 5, 7]\n    return sum(prices) * 100 + len(prices) * 10 + max(prices)\n", {'a': 2}),
        ("def solution(i):\n    xs = [10, 20, 30]\n    return xs[i] + xs[0] + xs[-1] + xs[-i]\n", {'i': 1}),
        ("def solution(i):\n    xs = [1, 2, 3]\n    xs[1] = 7\n    xs[i] += 10\n    xs[-1] = xs[0] * 2\n    return sum(xs)\n", {'i': 0}),
        ("def solution():\n    t = 0\n    for p in [3, 9, 4, 12]:\n        if p > 10:\n            break\n        if p == 9:\n            continue\n        t += p\n    return t\n", {}),
        ("def solution():\n    xs = [1, 2]\n    xs = [xs[1], xs[0]]\n    return xs[0] * 10 + xs[1]\n", {}),
        ("def solution():\n    t = 0\n    for a in [1, 2]:\n        for b in [10, 20]:\n            t += a * b\n    return t + min([4, 9, 2])\n", {}),
    ]
    for src, inputs in cases:
        want = python_value(src, inputs)
        for numbers in ('binary', 'rational'):
            r = score_python(src, inputs, {'x': str(want)}, N=32, numbers=numbers)
            assert r['status'] == 'ok' and r['value'] == want, (src, numbers, want, r['value'], r['fault_detail'])
    exact = score_python("def solution(i):\n    w = [0.5, 1.25, 2]\n    w[i] = w[i] * 3\n    return sum(w) / len(w)\n",
                         {'i': 1}, {'x': '25/12'}, N=32, numbers='rational')
    assert exact['status'] == 'ok'
    oob = score_python("def solution(i):\n    xs = [1, 2, 3]\n    return xs[i]\n", {'i': 3}, {'x': '0'}, N=32)
    assert oob['abstain_reason'] == 'runtime_error'                  # Python: IndexError
    for src in ("def solution():\n    xs = [1, 2]\n    ys = xs\n    return 0\n",
                "def solution():\n    xs = [1, 2]\n    xs.append(3)\n    return 0\n",
                "def solution():\n    xs = [1, 2]\n    xs = [1, 2, 3]\n    return 0\n",
                "def helper(v):\n    return 0\n\ndef solution():\n    xs = [1, 2]\n    return helper(xs)\n"):
        assert score_python(src, {}, {'x': '0'}, N=32)['abstain_reason'] == 'compile_error', src


def test_random_list_programs_match_cpython():
    """Lists mixed into random programs, both number modes. A loop over a
    list that its body both rebinds and changes is refused (that needs list
    identity, which one-variable-per-element cannot model)."""
    rng = random.Random(21)
    checked = 0
    for _ in range(150):
        src = Gen(rng, lists=True).program()
        inputs = {k: rng.randint(-30, 30) for k in 'abc'}
        try:
            want = python_value(src, inputs)
        except (OverflowError, ValueError):
            continue
        if abs(want) >= 2 ** 62:
            continue
        for numbers in ('binary', 'rational'):
            r = score_python(src, inputs, {'x': str(want)}, N=N, numbers=numbers)
            if r['abstain_reason'] == 'overflow' or (r['abstain_reason'] == 'compile_error' and (
                    'rebound and changed' in r['fault_detail'] or 'Out of memory' in r['fault_detail'])):
                continue
            assert r['status'] == 'ok' and r['value'] == want, (src, inputs, numbers, want, r['value'], r['fault_detail'])
            checked += 1
    assert checked > 150
    # the semantics that forced the snapshot: rebinding inside the loop does not change the iteration
    src = "def solution():\n    xs = [1, 2, 3]\n    t = 0\n    for v in xs:\n        t = t * 10 + v\n        xs = [7, 7, 7]\n    return t\n"
    assert score_python(src, {}, {'x': '123'}, N=N)['status'] == 'ok'
    src = "def solution():\n    xs = [1, 2, 3]\n    t = 0\n    for v in xs:\n        t = t * 10 + v\n        xs[2] = 9\n    return t\n"
    assert score_python(src, {}, {'x': '129'}, N=N)['status'] == 'ok'    # a store is seen later


def test_compile_workers_change_nothing():
    """compile_workers translates and compiles on worker processes; the
    results (including errors) must be the same as compiling inline."""
    items = [{'src': f"def solution(a):\n    t{k} = a * {k}\n    return t{k} / 4\n", 'inputs': {'a': 6},
              'option_texts': {'A': str(6 * k / 4)}} for k in range(6)]
    items += [{'src': "def solution(:\n", 'inputs': {}, 'option_texts': {'A': '1'}},
              {'src': "def solution(a):\n    return a ** a\n", 'inputs': {'a': 2}, 'option_texts': {'A': '4'}}]
    strip = lambda r: {k: v for k, v in r.items() if k != 'latency_s'}
    inline = [strip(r) for r in score_python_batch(items, N=32, numbers='rational')]
    pooled = [strip(r) for r in score_python_batch([dict(it, src=it['src'] + "\n") for it in items], N=32,
                                                   numbers='rational', compile_workers=2)]
    for a, b in zip(inline, pooled):
        a.pop('program_sha256'), b.pop('program_sha256')
        assert a == b


def test_round_digits_ties_away_from_zero():
    """round(x, n) rounds ties away from zero; round(x)
    keeps Python's half to even. Exact mode holds every decimal exactly, so
    -1.005 is a tie there; binary mode holds only binary fractions exactly
    (0.125, 1157.625), and like CPython's floats it sees -1.005 as just
    above the tie."""
    exact_ties = (("def solution():\n    return round(1157.625, 2)\n", '1157.63'),
                  ("def solution():\n    return round(0.125, 2)\n", '0.13'),
                  ("def solution():\n    return round(-0.125, 2)\n", '-0.13'),
                  ("def solution():\n    return round(2.5) * 10 + round(-2.5) + round(3.5) * 100\n", '418'))
    decimal_ties = (("def solution():\n    return round(-1.005, 2)\n", '-1.01'),
                    ("def solution(v):\n    return round(v * 1.05 ** 3, 2)\n", '1157.63'))
    for numbers, cases in (('rational', exact_ties + decimal_ties), ('binary', exact_ties)):
        for src, want in cases:
            r = score_python(src, {'v': 1000} if 'v' in src else {}, {'A': want}, N=32, numbers=numbers,
                             rounding_slack=1e-6 if numbers == 'binary' else None)
            assert r['status'] == 'ok', (numbers, src, r['value'], want)


def test_runner_code_and_inner_imports():
    """Ignore runner code around solution() and allow imports inside it.

    A function still cannot read a value produced by a top-level call.
    """
    cases = [
        "def solution():\n    import math\n    return math.ceil(7 / 2)\n\nprint(solution())\n",
        "def solution():\n    from math import floor\n    return floor(7 / 2) + 1\n\nif __name__ == '__main__':\n    print(solution())\n",
        "def solution():\n    return 4\n\nanswer = solution()\nprint(answer)\n",
    ]
    for src in cases:
        r = score_python(src, {}, {'x': '4'}, N=32, numbers='rational')
        assert r['status'] == 'ok', (src, r['abstain_reason'], r['fault_detail'])
    used = "def helper():\n    return base * 2\n\nbase = int('3')\n\ndef solution():\n    return helper()\n"
    assert score_python(used, {}, {'x': '6'}, N=32)['abstain_reason'] == 'compile_error'


def test_exponent_literals_and_none():
    r = score_python("import math\n\ndef solution():\n    return math.floor(9 / 4 + 1e-3) + 2.5e2\n", {}, {'x': '252'},
                     N=32, numbers='rational')
    assert r['status'] == 'ok', (r['abstain_reason'], r['fault_detail'])
    r = score_python("def solution():\n    return None\n", {}, {'x': '0'}, N=32)
    assert r['abstain_reason'] == 'compile_error' and 'None' in r['fault_detail']


def test_helper_parameters_take_the_type_of_their_arguments():
    """A helper called with a fractional argument gets a fractional
    parameter (it used to be typed int and truncated: 2.5 became 2)."""
    progs = [("def f(a, b, c):\n    return a * b * (c * 6)\n\ndef solution():\n    return f(5, 500, 2.5)\n", 37500),
             ("def g(a, b):\n    return int(a + b)\n\ndef h(a, b, c):\n    return a * b + c\n\n"
              "def solution():\n    return g(h(0.8, 10, 10), h(1, 2, 0.25))\n", 20),
             ("def pct(x, p):\n    return x * p / 100\n\ndef solution():\n    return pct(80, 12.5) + pct(3, 50)\n", 11.5)]
    for numbers in ('binary', 'rational'):
        for src, want in progs:
            ns = {}
            exec(src, ns)
            assert ns['solution']() == want
            r = score_python(src, {}, {'A': str(want)}, N=32, numbers=numbers)
            assert r['status'] == 'ok', (numbers, src, r['value'])



def test_python_names_never_share_a_c_name():
    """Keep generated temporaries, comprehension names, and user names separate.

    Regressions include t2, g1_acc/g1_i, and flattened names a__b.c vs a.b__c.
    """
    progs = [("def solution():\n    v = 600\n    {t} = 600 / (v - 200)\n    return {t}\n", '1.5'),
             ("def solution():\n    a = 5\n    b = 3\n    {t} = (a * 3) / (b + 1) + 1\n    return {t}\n", '4.75'),
             ("def solution():\n    g1_acc = 100\n    s = sum(i * 2 for i in range(4))\n    return s + g1_acc\n", '112'),
             ("def solution():\n    g1_i = 100\n    s = sum(i * 2 for i in range(4))\n    return s + g1_i\n", '112'),
             ("def a__b():\n    c = 1\n    return c\n\ndef a():\n    b__c = 10\n    v = a__b()\n    return b__c + v\n\n"
              "def solution():\n    return a()\n", '11')]
    for src, want in progs:
        for t in ('t1', 't2', 't3', 't4'):
            prog = src.replace('{t}', t)
            ns = {}
            exec(prog, ns)
            assert ns['solution']() == float(want)
            for numbers in ('binary', 'rational'):
                r = score_python(prog, {}, {'A': want}, N=64 if numbers == 'rational' else 32, numbers=numbers,
                                 rounding_slack=1e-6 if numbers == 'binary' else None)
                assert r['status'] == 'ok', (numbers, prog, r['value'])


ROOT_CASES = [      # (program, option, status, abstain_reason): exact roots, or the run stops
    ("import math\ndef solution():\n    return math.sqrt(144)\n", '12', 'ok', None),
    ("import math\ndef solution():\n    return math.sqrt(2)\n", '1.4142', 'abstain', 'inexact_sqrt'),
    ("import math\ndef solution():\n    return math.sqrt(-4)\n", '2', 'abstain', 'runtime_error'),
    ("import math\ndef solution():\n    return math.isqrt(1000)\n", '31', 'ok', None),
    ("import math\ndef solution():\n    return math.isqrt(-1)\n", '0', 'abstain', 'runtime_error'),
    ("def solution():\n    return 5625 ** 0.5\n", '75', 'ok', None),
    ("def solution():\n    return (9 ** 1.5 + 7) / 2\n", '17', 'ok', None),
    ("def solution():\n    return 12960000 ** 0.25\n", '60', 'ok', None),
    ("def solution():\n    return 27 ** (1/3) + 8 ** (2/3)\n", '7', 'ok', None),
    ("def solution():\n    x = 0.25\n    return x ** -0.5\n", '2', 'ok', None),
    ("def solution():\n    return (16 / 81) ** 0.25\n", '2/3', 'ok', None),
    ("def solution():\n    return 2 ** 0.5\n", '1.4142', 'abstain', 'inexact_sqrt'),
    ("def solution():\n    return 9007199254740993 ** 0.5\n", '94906265.6', 'abstain', 'inexact_sqrt'),
    ("def solution():\n    return (-8) ** (1/3)\n", '-2', 'abstain', 'runtime_error'),
    ("import math\ndef solution():\n    d = 7 ** 2 - 4 * 1 * 12\n    return (-7 + math.sqrt(d)) / 2\n", '-3', 'ok', None),
    ("def solution():\n    return 10000 * (1 + 0.08 / 4) ** (6 / 3)\n", '10404', 'ok', None),
    ("def solution():\n    return 2 ** -2 + 3 ** 2.0\n", '9.25', 'ok', None),
    ("def solution():\n    x = 0\n    return x ** -1\n", '0', 'abstain', 'runtime_error'),
]


def test_roots_are_exact_or_stop():
    """sqrt and fractional powers require exact rational roots; isqrt floors.
    An irrational root abstains as inexact_sqrt, and a negative base is a
    math domain error."""
    rs = score_python_batch([{'src': src, 'inputs': {}, 'option_texts': {'A': opt}} for src, opt, _, _ in ROOT_CASES],
                            N=64, numbers='rational')
    for (src, opt, status, reason), r in zip(ROOT_CASES, rs):
        assert (r['status'], r['abstain_reason']) == (status, reason), (src, r)
    r = score_python("def solution():\n    return 4 ** 0.5\n", {}, {'A': '2'}, N=32)        # binary fix: refused
    assert r['abstain_reason'] == 'compile_error' and 'rational' in r['fault_detail']


POW_CASES = [       # x ** n with an int n known only at run time: exact, or the run stops
    ("def solution():\n    p = 795\n    r = 8\n    n = 2\n    y = 2\n    return round(p * (1 + r / 100 / n) ** (n * y), 2)\n",
     '930.04', 'ok', None),
    ("def solution():\n    n = 10\n    return 2 ** n % 7\n", '2', 'ok', None),
    ("def solution():\n    n = -2\n    x = 0.5\n    return x ** n\n", '4', 'ok', None),
    ("def solution():\n    n = 0\n    x = 0.0\n    return x ** n\n", '1', 'ok', None),
    ("def solution():\n    total = 0\n    for k in range(1, 6):\n        total += 2 ** k\n    return total\n", '62', 'ok', None),
    ("def solution():\n    n = -2\n    return 2 ** n\n", '0.25', 'abstain', 'runtime_error'),       # an int base needs n >= 0
    ("def solution():\n    n = 5000\n    return 1 ** n\n", '1', 'abstain', 'step_limit'),
    ("def solution():\n    n = -1\n    x = 0.0\n    return x ** n\n", '0', 'abstain', 'runtime_error'),
    ("def solution():\n    n = 70\n    return 3 ** n\n", '1', 'abstain', 'overflow'),
]


def test_powers_known_at_run_time():
    """x ** n for an int n known only at run time (numbers='rational'): exact
    repeated multiplication; a fix base takes n < 0 as the reciprocal, an int
    base stops there (Python would give a float), |n| > 4096 stops as the step
    limit, and a fractional exponent known only at run time is refused."""
    rs = score_python_batch([{'src': src, 'inputs': {}, 'option_texts': {'A': opt}} for src, opt, _, _ in POW_CASES],
                            N=64, numbers='rational')
    for (src, opt, status, reason), r in zip(POW_CASES, rs):
        assert (r['status'], r['abstain_reason']) == (status, reason), (src, r)
    r = score_python("def solution():\n    t = 2.5\n    return 2 ** t\n", {}, {'A': '1'}, N=64, numbers='rational')
    assert r['abstain_reason'] == 'compile_error'


def test_floor_division_and_modulo_of_fix_values():
    """// and % with a fix operand follow Python's float semantics on the exact
    values: floor(a / b), and a - b * floor(a / b) with the sign of b; the
    result is a fix. Decimals are exact, so 0.7 // 0.1 is 7 (Python's binary
    floats give 6.0)."""
    from fractions import Fraction
    cases = ["7.5 // 2", "7.5 % 2", "-7.5 // 2", "-7.5 % 2", "7.5 // -2", "7.5 % -2", "10 // 0.25", "10 % 0.25",
             "0.7 // 0.1", "0.7 % 0.1", "100 % 3.0", "(17 * 1.5) % 4"]

    def exact(expr):
        node = ast.parse(expr, mode='eval').body
        val = lambda n: (-val(n.operand) if isinstance(n, ast.UnaryOp) else
                         val(n.left) * val(n.right) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) else
                         Fraction(repr(n.value)))
        a, b = val(node.left), val(node.right)
        q = (a / b).__floor__()
        return q if isinstance(node.op, ast.FloorDiv) else a - b * q
    items = [{'src': f"def solution():\n    return {c}\n", 'inputs': {}, 'option_texts': {'A': str(exact(c))}}
             for c in cases]
    for c, r in zip(cases, score_python_batch(items, N=64, numbers='rational')):
        assert r['status'] == 'ok', (c, r)
    r = score_python("def solution():\n    return 5.0 % 0\n", {}, {'A': '0'}, N=64, numbers='rational')
    assert r['abstain_reason'] == 'runtime_error'
