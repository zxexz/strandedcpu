# Python API

```python
from strandedcpu import (
    run_function, score_options, score_options_batch,
    score_python, score_python_batch,
)
```

## `strandedcpu.score_options`

```text
score_options(program, inputs, option_texts, *, N,
              result="return", backend="interpreter", s=64, m=128,
              max_steps=1_000_000, rounding_slack=None,
              numbers="binary", boolean_options=False)
```

Compile and run checked C, then compare the result with numeric options. `option_texts` maps labels to strings. Labels and their order do not affect matching. `result="return"` selects `main`'s integer return value; select a named `fix` global for fractions.

Inputs must name scalar globals. Integer inputs must be Python `int` values within the signed `N`-bit range; booleans are refused. Binary `fix` inputs must be exactly representable with `N` fractional bits and fit the two-word value. Rational inputs require `max(abs(numerator), denominator) < 2^(N-1)`.

Checks cover user-level integer and `fix` overflow, division by zero, array bounds, and shift amounts. Modular builtins such as `umulhi`, `qdiv`, `mulmod`, and `logadd`, and runtime-library internals, are not checked in the same way.

### Result fields

| Field | Meaning |
|---|---|
| `status` | `"ok"` when exactly one option matches; otherwise `"abstain"` |
| `abstain_reason`, `fault_detail` | Failure category and, when available, the check or input that caused it |
| `value` | Selected result as an integer or float; matching itself uses exact fractions |
| `value_raw` | Two stored words for a selected `fix` global; otherwise `None` |
| `parsed` | Parsed option values as integers/floats, or `None` for an unparseable string |
| `match` | Per-key booleans after comparison; it can be empty if execution or parsing failed |
| `match_kind` | `"exact"` or `"within_slack"` on a successful match; otherwise `None` |
| `steps`, `latency_s` | Instructions executed and wall-clock time |
| `inexact_ops` | Counts named `int_div_remainder` and `fix_rounding` |
| `backend`, `N` | Actual execution backend and word size |
| `program_sha256` | SHA-256 of the supplied program |
| `strandedcpu_commit`, `strandedcpu_diff_sha256` | Repository revision and hash of uncommitted state, when available |
| `fallback_reason` | Present when automatic backend selection or a word-size limit caused fallback |

`int_div_remainder` counts integer division that drops a fraction; `%` does not increment it. `fix_rounding` counts rounded binary literals/results and casts that drop a fraction. Neither counter forces abstention.

### Abstention reasons

| Reason | Trigger |
|---|---|
| `compile_error` | Unsupported or invalid source, unavailable global, or insufficient compilation resources |
| `input_out_of_domain` | An input cannot be represented in the declared type |
| `overflow` | Checked arithmetic exceeds its representable range |
| `runtime_error` | Division by zero, invalid array index or shift, negative root, or another checked domain error |
| `step_limit` | No halt within the limit, or a runtime exponent exceeds its supported limit |
| `inexact_sqrt` | A requested exact root is irrational |
| `unparseable_options` | At least one option cannot be parsed |
| `no_match` | No option equals the result |
| `multi_match` | More than one option matches |

### Option parsing and rounding

`parse_option_number(text)` returns an `int`, an exact `Fraction`, or `None`. It accepts signs, a leading `$`, correctly grouped thousands commas, decimals and simple fractions: `"$1,200"`, `"2.5"`, `"3/4"`. It refuses units, percentages, malformed commas, zero denominators and decimal fractions such as `"1.5/2"`.

`boolean_options=True` applies when every option parses as True/False or Yes/No, ignoring case and a final period. They become 1/0, and only an execution result of exactly 0 or 1 matches. `parse_option_truth(text)` exposes this parser.

Matching is exact by default. If `rounding_slack` is supplied and `fix_rounding > 0`, every option within that distance matches. Two nearby options still cause `multi_match`. Successful slack matching reports `match_kind="within_slack"`. Rational arithmetic is exact on parsed fraction values; it does not recover digits lost during [float-literal parsing](compiler.md#number-modes).

### Backend values

| Value | Execution |
|---|---|
| `"interpreter"` | CPU ISA interpreter; default |
| `"transformer"` | CPU interpreter run plus a CPU transformer comparison of metadata variables; use `N <= 16` |
| `"cuda"` | CUDA ISA kernel, one thread per program |
| `"auto"` | CUDA when a usable device and kernel are available; otherwise interpreter with `fallback_reason` |
| `"cuda-transformer"` | CUDA kernel run plus a CUDA transformer comparison of metadata variables; requires `N <= 16` |

Word widths extend through `N=64` on the interpreter and rational CUDA path. Binary-mode CUDA scoring above `N=32` falls back to the interpreter, even for integer-only source. Use `N <= 16` for the transformer.

The audit backends compare final variables listed in compiler metadata, excluding `__retval` and names beginning with `__`. They do not compare instruction counts or all machine state. Disagreement raises `RuntimeError`; a return-only C program has no variables to compare. The CPU audit accepts `N > 16` despite its precision limit; `cuda-transformer` rejects it.

## Batch scoring

```text
score_options_batch(calls, **common)
```

Calls contain `program`, `inputs`, `option_texts`, and optional keyword overrides. Per-call values override `common`; results retain input order. CUDA groups calls by `s`, `m`, `N`, step limit, and log scale for shared launches.

`compile_workers=k` with `k > 1` uses forkserver processes for compilation. Workers do not use CUDA. Put such calls behind an `if __name__ == "__main__":` guard.

## Python front end

```text
python_to_c(src, entry=None)
score_python(src, inputs, option_texts, *, N, entry=None, m=None, **kw)
score_python_batch(items, *, N, entry=None, m=None, **kw)
```

Import `python_to_c` from `strandedcpu.python`; scoring functions are exported at package level. Translation returns `(c_source, entry, params, result_is_fix)`. Parameters become C globals and the return value goes into global `result`. The default entry is `solution`, or the sole function. Types are inferred from source, not from the values supplied in `inputs`; entry parameters default to integer unless source inference promotes them.

Scoring passes keywords through to `score_options` and adds `c_source` to its result. A translation error returns `compile_error` with `c_source=None` and no repository revision fields. `program_sha256` identifies the Python source. Batch items contain `src`, `inputs`, `option_texts`, and optional `entry`. Default `m` is 512 for rational mode and 256 for binary mode.

See [Supported Python](compiler.md#python-subset) for syntax and arithmetic semantics.

## `strandedcpu.run_function`

```text
run_function(source, inputs=None, N=16, backend="interpreter",
             s=64, m=128, max_steps=1_000_000, trace=False,
             numbers="binary", device="cpu")
```

Compile C and return named variables. Declare input globals without initializers, because compiled initializers execute after the host writes inputs. `inputs` maps global names to integers, numbers for `fix`, or lists for integer arrays. Unknown names raise `KeyError`.

`backend` accepts `"interpreter"` or `"transformer"`. `device` selects the transformer's PyTorch device, including `"cuda"`. `N` is bits per word, `s` is scratchpad columns and the indirect-address window, and `m` is memory words. Compiled programs are cached by source and configuration; weights are cached by shape and device.

The result contains integers, integer arrays as lists, and `fix` values as floats in binary mode or `fractions.Fraction` in rational mode. It also contains `__retval` (the integer return slot) and `__steps`. With `trace=True`, `__trace` contains one entry per instruction:

```text
{"pc": ..., "op": ..., "b": ..., "c": ...,
 "writes": {memory_word: (old_value, new_value)}}
```

The trace keeps `pc` as a column address and subtracts `s` from operand addresses; `writes` uses memory offsets. A run that exceeds `max_steps` raises `RuntimeError`. This function runs without the arithmetic checks used by option scoring; ordinary word arithmetic can wrap.

## Transformer state API

Import these from `strandedcpu.transformer` when working directly with instructions and tensors:

| Interface | Behavior |
|---|---|
| `Config(s=8, m=8, n=32, N=8, lam=10.0, log_frac_bits=2, vector_unit=True)` | Defines scratchpad, memory, total columns, word width and extension settings; derives `d_model`, `logn`, `page2` and `vector` |
| `Computer(cfg)` | Builds program-independent transformer weights |
| `init_state(cfg, memory, commands)` | Packs memory and instruction triples into a state tensor of shape `(cfg.d_model, cfg.n)` |
| `computer.step(state)` | Returns the state after one instruction through eight layers |
| `computer.to(device)` | Moves transformer weights to a PyTorch device; move the state tensor too |
| `read_memory(state, cfg)` | Decodes the `m` signed memory words |
| `get_pc(state, cfg)` | Decodes the PC column address; 0 indicates halt |

`init_state` truncates memory after `m` words and instructions after the available columns. Ensure `s + m + len(commands) <= n` before packing a program. Use `torch.no_grad()` for execution. See the [transformer tutorial](../tutorials/first-program.md).

## Decision-model features

```text
strandedcpu.decision.score_row(items, *, N=32, numbers="rational",
                   boolean_options=True, max_steps=1_000_000,
                   backend="interpreter", **kw)
strandedcpu.decision.score_item(program, inputs, options, *, language="python", **kw)
strandedcpu.decision.option_features(result, option_keys)
```

A row item contains `program`, `options`, optional `inputs`, and `language` (`"python"` by default, or `"c"`). `score_row` returns `{"items": [{"result": ..., "features": ...}, ...], "latency_s": ...}` in input order. `score_item` returns one such item. Read `strandedcpu.decision.FEATURES` for the exact vector order:

1. `match`, `call_ok`, `other_matched`.
2. `abstain_compile_error`, `abstain_input_out_of_domain`, `abstain_overflow`, `abstain_runtime_error`, `abstain_step_limit`, `abstain_unparseable_options`, `abstain_no_match`, `abstain_multi_match`.
3. `inexact_div`, `inexact_round`, `log2_steps`.

The first three flags describe successful calls: `match` marks the selected option, `call_ok` marks a unique match, and `other_matched` marks the remaining options of that successful call. Inexact flags indicate positive counts; `log2_steps` is `log2(1 + steps)`. The current feature list has no separate `inexact_sqrt` abstention flag; that reason still gives `call_ok=0` and remains in `result`.

## Model-generated programs

Import from [`strandedcpu.gemma`](../../strandedcpu/gemma.py).

```text
write_programs(model, tok, questions, batch_size=16, max_new_tokens=1024,
               temperature=0.0, samples=1, prompt=PROMPT)
extract_program(text)
check_programs(programs, golds, backend="auto", compile_workers=0,
               max_steps=10_000_000)
decisions(programs, options_list, backend="auto")
summarize(results)
```

`write_programs` expects a model with `generate()` and `device`, plus a tokenizer with chat templating, encoding, and decoding. It returns `samples` replies per question. Temperature zero is greedy; positive temperatures use `top_p=0.95`. It sets left padding and uses EOS for padding when necessary. Custom prompts need a `{question}` format field.

`extract_program` returns the last Python or unlabelled fenced block, stripped with a final newline, or `None`. `check_programs` scores against numeric gold texts with no function inputs, rational mode, and `N=32`. Missing programs produce `compile_error`. Results are full scoring dictionaries; mismatched sequence lengths raise `ValueError`.

`decisions` returns a uniquely matched label or `None`. Choices use labels such as `A)` and `B)` with numeric text. Roots and strings containing several numbers are excluded; no numeric candidates gives `None`. It uses rational mode, `N=32`, and a 10-million-step limit. Mismatched sequence lengths raise `ValueError`.

`summarize` returns `n`, `correct`, `accuracy`, and failure counts by abstention reason. It counts `status == "ok"` as correct, which means a gold match when used with `check_programs`.

## Host mailbox

```text
run_with_mailbox(source, batches, N=16, s=128, m=256,
                 backend="interpreter", max_steps=2_000_000, inputs=None)
```

Import from `strandedcpu.host`. Source must include `strandedcpu.runtime.mailbox(capacity)`. Each batch is a list of integers no longer than the mailbox capacity. When the program requests data with `mb_wait()`, the host fills `mb`, sets `mb_len` and clears `mb_req`. Exhausted input sets `mb_len=0`. Supported execution paths are the interpreter and transformer. The return value contains named variables and `__steps`.

## iCEstick hardware interfaces

`strandedcpu.icestick` loads and runs raw 8-bit programs. `strandedcpu.icestick_transformer` offloads selected transformer arithmetic. They require different FPGA images and are not values of the scoring API's `backend` argument. See [iCEstick reference](icestick.md).

## CUDA build controls

[`strandedcpu/cuda.py`](../../strandedcpu/cuda.py) first loads the packaged `kernel.fatbin`, after verifying its kernel-source hash against `kernel.fatbin.json`. The binary contains `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100` and `sm_120` machine code plus `compute_75` PTX.

If loading fails, build attempts continue in this order: NVRTC for the device, NVRTC PTX, nvcc through PyTorch's extension loader, and nvcc PTX. All paths compile the same kernel source. `STRANDEDCPU_CUDA_BUILD=prebuilt|nvrtc|nvrtc-ptx|nvcc|nvcc-ptx` forces one path. `strandedcpu.cuda.BUILD` records the chosen `mode` and earlier `errors`.

`python -m strandedcpu.cuda build-fatbin` rebuilds the shipped binary and metadata. It needs nvcc 12.8 or newer for the full target list, and no GPU. `tests/test_prebuilt.py` checks that source and binary metadata agree. Kernel launches use PyTorch's current CUDA stream.
