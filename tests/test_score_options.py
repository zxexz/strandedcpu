"""Test option parsing, matching, abstention, and backend agreement."""

import itertools
import random
from fractions import Fraction

from strandedcpu.api import parse_option_number, score_options, run_function


def facts(result):
    """The parts of a result that must not depend on option keys."""
    return (result['status'], result['abstain_reason'], result['fault_detail'], result['value'],
            result['value_raw'], result['inexact_ops'], result['steps'])


# parse_option_number

def test_parse_accepts():
    cases = {
        '1200': 1200, '$1,200': 1200, '-3': -3, '+7': 7, '2.5': Fraction(5, 2), '1200.00': 1200,
        ' 42 ': 42, '1/2': Fraction(1, 2), '-3/4': Fraction(-3, 4), '-$5': -5,
        '1,234,567': 1234567, '0': 0, '-0': 0, '0.125': Fraction(1, 8), '007': 7,
    }
    for text, want in cases.items():
        got = parse_option_number(text)
        assert got == want and type(got) is type(want), (text, got)


def test_parse_rejects():
    for text in ['12 apples', '50%', '1,20', '1,2000', '12,34', 'abc', '', ' ', '3/0', '1.5/2',
                 '$-5', '1e3', '0x10', '5.', '.5', '1 000', '−5', 'five', '1,200/3', None, 5]:
        assert parse_option_number(text) is None, text


# score_options: statuses

SQUARE_PLUS = "int a; int b; int main() { return a * b + 7; }"


def test_ok_unique_match():
    r = score_options(SQUARE_PLUS, {'a': 30, 'b': 40}, {'x': '1,207', 'y': '1200', 'z': '$1,217'}, N=16)
    assert r['status'] == 'ok' and r['abstain_reason'] is None
    assert r['value'] == 1207 and r['match'] == {'x': True, 'y': False, 'z': False}
    assert r['parsed'] == {'x': 1207, 'y': 1200, 'z': 1217}
    assert len(r['program_sha256']) == 64 and r['strandedcpu_commit'] and r['latency_s'] >= 0 and r['steps'] > 0


def test_gsm8k_distractors_separate():
    gold = 1236
    opts = {k: str(v) for k, v in zip('abcdefghij', [gold, gold + 1, gold - 1, gold + 2, gold - 2,
                                                       gold + 10, gold * 2, gold * 10, gold // 2, gold // 10])}
    r = score_options("int x; int main() { return x * 4 + 36; }", {'x': 300}, opts, N=16)
    assert r['status'] == 'ok' and [k for k, v in r['match'].items() if v] == ['a']


def test_abstain_reasons():
    def reason(program, inputs, options, **kw):
        r = score_options(program, inputs, options, N=16, **kw)
        assert r['status'] == 'abstain'
        return r['abstain_reason'], r['fault_detail']

    assert reason("int a; int main() { return a + ; }", {'a': 1}, {'p': '1'})[0] == 'compile_error'
    assert reason("int a; int main() { return a; }", {'nope': 1}, {'p': '1'})[0] == 'compile_error'
    assert reason("int a; int main() { return a; }", {'a': 99999}, {'p': '1'})[0] == 'input_out_of_domain'
    assert reason("int a; int main() { return a; }", {'a': 1.5}, {'p': '1'})[0] == 'input_out_of_domain'
    assert reason(SQUARE_PLUS, {'a': 300, 'b': 400}, {'p': '120007'}) == ('overflow', 'mul: 300 and 400 give 120000')
    assert reason("int a; int main() { return 40000; }", {}, {'p': '40000'})[0] == 'overflow'
    assert reason("int a; int main() { return a / 0; }", {'a': 3}, {'p': '0'}) == ('runtime_error', 'division_by_zero')
    assert reason("int a; int t[4]; int main() { return t[a]; }", {'a': 4}, {'p': '0'})[1].startswith('index_out_of_bounds')
    assert reason("int a; int main() { return 1 << a; }", {'a': 16}, {'p': '0'})[1].startswith('shift_amount')
    assert reason("int a; int main() { while (1) { a += 0; } return a; }", {'a': 1}, {'p': '1'},
                  max_steps=300)[0] == 'step_limit'
    assert reason("int a; int main() { return a; }", {'a': 5}, {'p': '5', 'q': '5.0'})[0] == 'multi_match'
    assert reason("int a; int main() { return a; }", {'a': 5}, {'p': '4', 'q': '6'})[0] == 'no_match'
    assert reason("int a; int main() { return a; }", {'a': 5}, {'p': '5 apples', 'q': '6'})[0] == 'unparseable_options'


def test_intermediate_overflow_is_caught():
    """a*b overflows although (a*b)/c would fit: a wrapped intermediate must
    not produce an in-range, exact-looking answer."""
    r = score_options("int a; int b; int c; int main() { return a * b / c; }",
                      {'a': 300, 'b': 300, 'c': 100}, {'p': '900'}, N=16)
    assert (r['status'], r['abstain_reason']) == ('abstain', 'overflow')
    for program in ("int a; int main() { return -a; }", "int a; int main() { return abs(a); }",
                    "int a; int main() { a -= 1; return a; }", "int a; int main() { return a << 1; }"):
        r = score_options(program, {'a': -32768}, {'p': '0'}, N=16)
        assert r['abstain_reason'] == 'overflow', program


def test_inexact_ops_counted_not_abstained():
    r = score_options("int a; int main() { return a / 2; }", {'a': 7}, {'p': '3', 'q': '4'}, N=16)
    assert r['status'] == 'ok' and r['value'] == 3 and r['inexact_ops'] == {'int_div_remainder': 1, 'fix_rounding': 0}
    r = score_options("int a; int main() { return a / 2; }", {'a': 8}, {'p': '4'}, N=16)
    assert r['inexact_ops'] == {'int_div_remainder': 0, 'fix_rounding': 0}
    r = score_options("fix p; fix q; int main() { q = p / 3; return 0; }", {'p': 1}, {'p': '1/3'}, N=16, result='q')
    assert r['status'] == 'abstain' and r['abstain_reason'] == 'no_match' and r['inexact_ops']['fix_rounding'] == 1
    r = score_options("fix p; fix q; int main() { q = p / 2; return 0; }", {'p': 7}, {'a': '3', 'b': '3.5'}, N=16, result='q')
    assert r['status'] == 'ok' and r['match'] == {'a': False, 'b': True} and r['value_raw'] == (3, -32768)
    r = score_options("fix p; int main() { p = 0.1 * 3; return 0; }", {}, {'a': '0.3'}, N=16, result='p')
    assert r['inexact_ops']['fix_rounding'] >= 1


def test_comparisons_exact_at_extremes():
    """<, <=, >, >= used to test the sign of a - b, which wraps."""
    src = "int a; int b; int lt; int le; int gt; int ge; int main() { lt = a < b; le = a <= b; gt = a > b; ge = a >= b; return 0; }"
    for a, b in [(30000, -30000), (-30000, 30000), (32767, -32768), (-32768, 32767), (5, 5), (-1, 0)]:
        o = run_function(src, {'a': a, 'b': b}, N=16)
        assert (o['lt'], o['le'], o['gt'], o['ge']) == (int(a < b), int(a <= b), int(a > b), int(a >= b)), (a, b)


# Keys are opaque

def test_invariant_to_key_permutation_and_renaming():
    rng = random.Random(0)
    program = "int x; int main() { return x * 3 - 4; }"
    texts = ['53', '55', '$530', '26', '56', '5.3']
    for x in (19, 20, 3):
        base = score_options(program, {'x': x}, {f'k{i}': t for i, t in enumerate(texts)}, N=16)
        per_text = {texts[i]: base['match'].get(f'k{i}') for i in range(len(texts))}
        for _ in range(10):
            order = texts[:]
            rng.shuffle(order)
            keys = rng.sample(['a', 'b', 'c', 'd', 'e', 'f', 'ZZ', '0', 'option 1', ''], len(order))
            r = score_options(program, {'x': x}, dict(zip(keys, order)), N=16)
            assert facts(r) == facts(base)
            assert {t: r['match'].get(k) for k, t in zip(keys, order)} == per_text


# Backends agree

def test_transformer_backend_agrees():
    r = score_options("int x; int main() { return x * x - 3; }", {'x': 21}, {'a': '438', 'b': '441'},
                      N=16, backend='transformer', s=32, m=32)
    assert r['status'] == 'ok' and r['match'] == {'a': True, 'b': False} and r['backend'] == 'transformer'


def test_deterministic():
    args = ("int x; int main() { return x / 3; }", {'x': 100}, {'a': '33', 'b': '34'})
    a, b = score_options(*args, N=16), score_options(*args, N=16)
    assert facts(a) == facts(b) and a['program_sha256'] == b['program_sha256'] and a['strandedcpu_commit'] == b['strandedcpu_commit']


def test_n32_interpreter():
    """GSM8K golds can exceed 32767; N=32 runs on the interpreter (the
    transformer is exact only while its threshold sums fit float32, N <= 16)."""
    r = score_options("int a; int b; int main() { return a * b + 7; }", {'a': 30000, 'b': 4000},
                      {'x': '120,000,007', 'y': '120000008', 'z': '1,200,000,070'}, N=32)
    assert r['status'] == 'ok' and r['match'] == {'x': True, 'y': False, 'z': False}
    r = score_options("int a; int b; int main() { return a * b; }", {'a': 70000, 'b': 70000}, {'x': '0'}, N=32)
    assert r['abstain_reason'] == 'overflow'
    src = "int a; int b; int lt; int main() { lt = a < b; return lt; }"
    assert [run_function(src, {'a': a, 'b': b}, N=32)['lt'] for a, b in
            [(2 ** 31 - 1, -2 ** 31), (-2 ** 31, 2 ** 31 - 1)]] == [0, 1]


# numbers='rational': exact fix arithmetic

def test_rational_is_exact_where_binary_rounds():
    prog = "fix price; fix r; int main() { r = price * 0.8; return 0; }"
    opts = {'A': '40', 'B': '41'}
    binary = score_options(prog, {'price': 50}, opts, N=32, result='r')
    exact = score_options(prog, {'price': 50}, opts, N=32, result='r', numbers='rational')
    assert binary['abstain_reason'] == 'no_match'
    assert exact['status'] == 'ok' and exact['match']['A'] and exact['match_kind'] == 'exact'
    assert exact['inexact_ops'] == {'int_div_remainder': 0, 'fix_rounding': 0}
    # binary rounding can cross an integer before a cast and match a distractor
    cast = "fix x; int r; int main() { r = (int) (x * 0.15 + 0.5); return 0; }"
    assert score_options(cast, {'x': 10}, {'A': '1', 'B': '2'}, N=32, result='r')['match']['A']
    assert score_options(cast, {'x': 10}, {'A': '1', 'B': '2'}, N=32, result='r', numbers='rational')['match']['B']


def test_rational_values_inputs_and_faults():
    third = score_options("fix x; fix r; int main() { r = x / 3 + 0.1 + 0.2; return 0; }", {'x': 10},
                          {'A': '109/30', 'B': '3.63'}, N=32, result='r', numbers='rational')
    assert third['status'] == 'ok' and third['match']['A'] and third['value_raw'] == (109, 30)
    decimal_in = score_options("fix x; fix r; int main() { r = x * 3; return 0; }", {'x': 0.1},
                               {'A': '0.3'}, N=32, result='r', numbers='rational')
    assert decimal_in['status'] == 'ok'                                 # 0.1 as written, not as a double
    floor = score_options("fix x; int r; int main() { r = (int) (x / 2); return 0; }", {'x': -7},
                          {'A': '-4'}, N=32, result='r', numbers='rational')
    assert floor['status'] == 'ok' and floor['inexact_ops']['fix_rounding'] == 1     # (int) dropped 1/2
    zero = score_options("fix x; fix r; int main() { r = x / 0; return 0; }", {'x': 1}, {'A': '0'},
                         N=32, result='r', numbers='rational')
    assert zero['abstain_reason'] == 'runtime_error' and 'division_by_zero' in zero['fault_detail']
    big = score_options("fix x; fix r; int main() { r = x / 3; r = r * x * x * x * x; return 0; }", {'x': 1000},
                        {'A': '0'}, N=32, result='r', numbers='rational')
    assert big['abstain_reason'] == 'overflow'
    out = score_options("fix x; int main() { return 0; }", {'x': 2 ** 40}, {'A': '0'}, N=32, numbers='rational')
    assert out['abstain_reason'] == 'input_out_of_domain'


def test_rational_transformer_agrees():
    prog = "fix x; fix r; fix h(fix v) { return v * v / 2; }\nint main() { r = h(x) - 0.25; return 0; }"
    r = score_options(prog, {'x': 1.5}, {'A': '0.875'}, N=16, result='r', numbers='rational',
                      backend='transformer', m=128)
    assert r['status'] == 'ok' and r['backend'] == 'transformer'


# boolean_options: True/False and Yes/No answers

def test_boolean_options():
    from strandedcpu.python import score_python
    rng = random.Random(4)

    def expr(d=0):          # BBH boolean_expressions style
        if d > 2 or rng.random() < 0.3:
            return rng.choice(['True', 'False'])
        k = rng.random()
        if k < 0.25:
            return f"not {expr(d + 1)}"
        if k < 0.45:
            return f"( {expr(d + 1)} )"
        return f"{expr(d + 1)} {rng.choice(['and', 'or'])} {expr(d + 1)}"

    for _ in range(150):
        e = expr()
        r = score_python(f"def solution():\n    return {e}\n", {}, {'A': 'True', 'B': 'False'}, N=16,
                         boolean_options=True)
        assert r['status'] == 'ok' and r['match']['A'] == eval(e), e
    yn = score_python("def solution(a):\n    return a > 3\n", {'a': 5}, {'A': 'Yes', 'B': 'no.'}, N=16,
                      boolean_options=True)
    assert yn['status'] == 'ok' and yn['match'] == {'A': True, 'B': False}
    # opt-in only; a result that is not 0/1 never matches; mixed options are not truth options
    assert score_python("def solution():\n    return True\n", {}, {'A': 'True', 'B': 'False'},
                        N=16)['abstain_reason'] == 'unparseable_options'
    assert score_python("def solution():\n    return 5\n", {}, {'A': 'True', 'B': 'False'}, N=16,
                        boolean_options=True)['abstain_reason'] == 'no_match'
    assert score_python("def solution():\n    return 1\n", {}, {'A': 'True', 'B': '2'}, N=16,
                        boolean_options=True)['abstain_reason'] == 'unparseable_options'


def test_auto_backend_falls_back_cleanly():
    """backend='auto' runs the CUDA kernel when a GPU is there, else the
    interpreter, and says why in fallback_reason. The facts are the same."""
    import torch
    r = score_options("int x; int main() { return x * 3; }", {'x': 7}, {'A': '21'}, N=32, backend='auto')
    assert r['status'] == 'ok'
    if torch.cuda.is_available():
        assert r['backend'] == 'cuda' and 'fallback_reason' not in r
    else:
        assert r['backend'] == 'interpreter' and r['fallback_reason']


def test_64_bit_rational_fits_what_32_bit_overflows():
    """A Bayes ratio with two-digit probabilities: its products outgrow 32
    bits but fit 64, exactly."""
    from strandedcpu.python import score_python
    bayes = ("def solution():\n    pz = 0.37\n    n1 = pz * 0.62 * 0.81 + (1 - pz) * 0.21 * 0.44\n"
             "    d1 = pz * 0.62 + (1 - pz) * 0.21\n    n0 = pz * 0.38 * 0.52 + (1 - pz) * 0.79 * 0.29\n"
             "    d0 = pz * 0.38 + (1 - pz) * 0.79\n    return n1 / d1 - n0 / d0\n")
    gold = {'A': '771119393/2308731100', 'B': '0.334'}
    assert score_python(bayes, {}, gold, N=32, numbers='rational')['abstain_reason'] == 'overflow'
    r = score_python(bayes, {}, gold, N=64, numbers='rational')
    assert r['status'] == 'ok' and r['match'] == {'A': True, 'B': False}


def test_binary_fix_above_32_bits_runs_on_the_interpreter():
    """backend='cuda' with binary fix at N > 32 runs on the interpreter and
    says why (the kernel's fix checks need 4N-bit products); no GPU needed."""
    from strandedcpu.api import score_options
    r = score_options("fix price; fix r; int main() { r = price * 0.75; return 0; }", {'price': 50}, {'A': '37.5'},
                      N=64, result='r', backend='cuda')
    assert r['status'] == 'ok' and r['backend'] == 'interpreter' and 'N=32' in r['fallback_reason']
