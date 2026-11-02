"""Test per-option feature vectors and rows of C/Python scoring requests."""

from strandedcpu.decision import FEATURES, score_item, score_row


def test_features_and_rows():
    row = score_row([
        {'program': "def solution(price):\n    return price * 0.8\n", 'inputs': {'price': 50},
         'options': {'A': '40', 'B': '41'}},
        {'program': "def solution():\n    return not True or False\n", 'options': {'A': 'True', 'B': 'False'}},
        {'program': "def solution():\n    return [1][5]\n", 'options': {'A': '1'}},
        {'program': "int x; int main() { return x * 3; }", 'language': 'c', 'inputs': {'x': 7},
         'options': {'A': '21', 'B': '22'}},
    ])
    assert row['latency_s'] > 0 and len(row['items']) == 4
    f = {name: i for i, name in enumerate(FEATURES)}
    a, b = row['items'][0]['features']['A'], row['items'][0]['features']['B']
    assert len(a) == len(FEATURES)
    assert (a[f['match']], a[f['call_ok']], b[f['match']], b[f['other_matched']]) == (1, 1, 0, 1)
    assert row['items'][1]['features']['B'][f['match']] == 1                    # not True or False -> False
    bad = row['items'][2]['features']['A']
    assert bad[f['call_ok']] == 0 and bad[f['abstain_compile_error']] == 1   # list literal indexed by a constant out of range
    assert row['items'][3]['features']['A'][f['match']] == 1
    # exact defaults: no rounding happened, so neither inexact flag is set
    assert a[f['inexact_round']] == 0 and a[f['inexact_div']] == 0


def test_abstain_reasons_are_one_hot():
    f = {name: i for i, name in enumerate(FEATURES)}
    r = score_item("def solution(a):\n    return a // 0\n", {'a': 1}, {'A': '0'})
    v = r['features']['A']
    assert v[f['abstain_runtime_error']] == 1
    assert sum(v[f[n]] for n in FEATURES if n.startswith('abstain_')) == 1
