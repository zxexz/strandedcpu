# Score answer options

Use `score_options` for C source or `score_python` for the [Python subset](../reference/compiler.md#python-subset). The examples below assume an [installed checkout](../tutorials/checked-python.md#install).

## Score a C result

Declare scalar input globals without initializers:

```python
from strandedcpu import score_options

program = """
int price;
int count;
int main() { return price * count; }
"""

scored = score_options(
    program,
    {"price": 12, "count": 4},
    {"A": "$48", "B": "$16", "C": "$40"},
    N=16,
)
assert scored["status"] == "ok"
assert scored["match"] == {"A": True, "B": False, "C": False}
```

Choose `N` to hold intermediate values as well as the final result. Checked arithmetic reports overflow rather than matching a wrapped value.

Select an option only on success:

```python
if scored["status"] == "ok":
    selected = next(key for key, matches in scored["match"].items() if matches)
    print(selected)
else:
    print(scored["abstain_reason"], scored["fault_detail"])
```

For `overflow`, inspect intermediates or increase `N`. For `no_match`, check the program and choices. For `multi_match`, check duplicate numeric choices or a supplied tolerance. The [API reference](../reference/api.md#abstention-reasons) lists the remaining reasons.

Options must parse as numbers, such as `"$1,200"`, `"2.5"`, or `"3/4"`. `"5 apples"` and `"25%"` are refused; one unparseable option causes abstention.

## Use exact fractions

Set `numbers="rational"`. C fractional results use a `fix` global selected by `result`, since `main` returns an integer:

```python
from strandedcpu import score_options

program = """
fix price;
fix total;
int main() { total = price * 0.8; return 0; }
"""

scored = score_options(
    program, {"price": 50}, {"A": "40", "B": "41"},
    N=32, result="total", numbers="rational",
)
assert scored["status"] == "ok"
assert scored["match"]["A"]
assert scored["value_raw"] == (40, 1)
```

The pair `(40, 1)` is the reduced numerator and denominator. Both components and intermediate calculations must fit the word width; `N=64` provides more room.

The default `numbers="binary"` uses fixed point. `50 * 0.8` can then fall near `40` without equaling it. If approximate matching is intended, set `rounding_slack`, for example `1e-6` at `N=32`, below the spacing between choices. Slack applies only when `inexact_ops["fix_rounding"] > 0`; a successful tolerant match has `match_kind == "within_slack"`.

Inexact-operation counts do not force abstention. For example, C integer `7 / 2` can match `"3"` while incrementing `int_div_remainder`. Inspect `inexact_ops` when that distinction matters.

## Batch work on CUDA

Use `backend="auto"` to try CUDA with interpreter fallback:

```python
scored = score_options(
    "int x; int main() { return x + 1; }",
    {"x": 5}, {"A": "6", "B": "7"},
    N=16, backend="auto",
)
print(scored["backend"])
print(scored.get("fallback_reason"))
```

Check the returned backend when measuring a run. Explicit `backend="cuda"` does not provide the general automatic fallback, though binary-mode scoring above `N=32` still redirects to the interpreter. Rational CUDA execution supports `N=64`.

Batch compatible programs to share a launch:

```python
from strandedcpu import score_options_batch

results = score_options_batch(
    [
        {"program": "int x; int main() { return x + 1; }",
         "inputs": {"x": 5}, "option_texts": {"A": "6", "B": "7"}},
        {"program": "int x; int main() { return x * 2; }",
         "inputs": {"x": 5}, "option_texts": {"A": "10", "B": "12"}},
    ],
    N=16, backend="auto",
)
assert all(item["status"] == "ok" for item in results)
```

For a transformer audit, use `backend="transformer"`, `N <= 16`, and a named output global selected by `result`. The audit compares named user variables and excludes the return slot. See [backend values](../reference/api.md#backend-values) for the comparison and device choices.
