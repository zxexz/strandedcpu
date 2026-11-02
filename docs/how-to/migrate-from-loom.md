# Migrate an older LOOM-based checkout

The top-level modules have moved into the `strandedcpu` package. Old imports and compatibility aliases have been removed. [Project scope](../explanation/project-scope.md) describes the relationship to LOOM.

## Imports and names

Install from the checkout:

```bash
python -m pip install -e .
```

Execution and scoring functions are exported at package level:

```python
from strandedcpu import (
    run_function, score_options, score_options_batch,
    score_python, score_python_batch,
)
```

| Old module | New module |
|---|---|
| `loom_api` | `strandedcpu.api` |
| `loom_py` | `strandedcpu.python` |
| `loom_jev` | `strandedcpu.decision` |
| `loom_cuda` | `strandedcpu.cuda` |
| `loom_gemma` | `strandedcpu.gemma` |
| `loom_host` | `strandedcpu.host` |
| `loom_runtime` | `strandedcpu.runtime` |
| `loom_v1` | `strandedcpu.transformer` |
| `loom_v1_standard` | `strandedcpu.standard` |
| `c_compiler` | `strandedcpu.compiler` |
| `isa_interpreter` | `strandedcpu.interpreter` |
| `bayes_ref` | `strandedcpu.reference` |
| `subleq` | `strandedcpu.subleq` |

Rename transformer `LoomConfig` and `LoomComputer` to `Config` and `Computer`. The standard variant uses `StandardConfig` and `StandardComputer`. `LoomFault` is now `ExecutionFault`; the Gemma helper `loom_decisions` is now `decisions`.

```python
from strandedcpu.transformer import Config, Computer, init_state, read_memory, get_pc
from strandedcpu.compiler import compile_c, ExecutionFault
```

The `extended_isa_v4*` shims are gone. Use the eight-layer transformer in place of the deprecated merged variant, which lacked attention needed by ARGMAX and vector operations.

## Environment and results

| Old name | New name |
|---|---|
| `LOOM_CUDA_BUILD` | `STRANDEDCPU_CUDA_BUILD` |
| `LOOM_FUZZ` | `STRANDEDCPU_FUZZ` |
| Result `loom_commit` | `strandedcpu_commit` |
| Result `loom_diff_sha256` | `strandedcpu_diff_sha256` |
| `loom_kernel.fatbin` and metadata | `strandedcpu/kernel.fatbin` and `strandedcpu/kernel.fatbin.json` |

Rebuild the CUDA binary with `python -m strandedcpu.cuda build-fatbin`. The internal CUDA symbol `loom_kernel` remains to preserve the binary/source-hash contract.

## Tests

Tests live under `tests/`. Most retain their filenames; these were also renamed:

| Old file | New file |
|---|---|
| `test_loom_py.py` | `tests/test_python.py` |
| `test_loom_cuda.py` | `tests/test_cuda.py` |
| `test_loom_jev.py` | `tests/test_decision.py` |
| `test_v4.py` | `tests/test_transformer.py` |
| `test_v4_standard.py` | `tests/test_standard.py` |

Run pytest against the directory and standalone runners as modules:

```bash
python -m pytest tests -q
python -m tests.test_transformer
python -m tests.test_standard
python -m tests.test_bayes_exhaustive
python -m tests.test_regression cpu --level 1
```

The regression runner covers levels 1–4; the ONNX level was removed. The RTL harness stays at `hardware/icestick/test_core.py`. See [Run the tests](run-tests.md) for dependencies.

## Removed assets

Browser demos, ONNX exports, game assets, and the Alveo FPGA tree remain available in [upstream LOOM](https://github.com/mkturkcan/Loom). They are no longer shipped here. The retained JavaScript interpreter moved from `demos/isa_interpreter.js` to [`javascript/interpreter.js`](../../javascript/interpreter.js); its comparison suite is `tests/test_js_interpreter.py`.

Current hardware work lives under `hardware/icestick/`. The [iCEstick guide](run-icestick.md) covers its two images.
