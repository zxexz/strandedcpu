# Check a Python answer

This example computes a discounted price on the CPU and compares it with three answer options.

## Install

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Compute the price

Save `checked_example.py`:

```python
from strandedcpu import score_python

program = """
def solution(price, count):
    return price * count * 0.8
"""

scored = score_python(
    program,
    {"price": 25, "count": 2},
    {"A": "$40", "B": "$41", "C": "$60"},
    N=32,
    numbers="rational",
)

print("status =", scored["status"])
print("value =", scored["value"])
print("matches =", scored["match"])
```

Run it:

```bash
.venv/bin/python checked_example.py
```

```text
status = ok
value = 40
matches = {'A': True, 'B': False, 'C': False}
```

`numbers="rational"` makes `0.8` exactly `4/5`. Each numerator and denominator must fit a signed `N`-bit word. The two input parameters here are integers.

The scorer parses `"$40"` as a number. Exactly one choice equals the computed result, so `status` is `"ok"` and `match["A"]` is true. Labels identify choices; changing their names does not change the comparison.

## Remove the matching option

Append this call:

```python
unmatched = score_python(
    program,
    {"price": 25, "count": 2},
    {"A": "$39", "B": "$41"},
    N=32,
    numbers="rational",
)
print("status =", unmatched["status"])
print("reason =", unmatched["abstain_reason"])
```

The new output is:

```text
status = abstain
reason = no_match
```

The value is still `40`; neither choice matches. A caller should select an option only when `status == "ok"`. On an abstention, `abstain_reason` and `fault_detail` describe the failure.

Change the first call's `count` to `3`. It should now return `60` and match option `C`.

[Compile C](compiled-c.md) next, or see [option scoring](../how-to/score-options.md) for batching and CUDA. The [Python subset reference](../reference/compiler.md#python-subset) lists supported syntax.
