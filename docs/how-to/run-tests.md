# Run the tests

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest tests -q
```

Use an existing environment's Python if the package and pytest are already installed.

## Standalone runners

The transformer-variant and regression runners have module entry points. The exhaustive arithmetic suite can also run this way:

```bash
.venv/bin/python -m tests.test_transformer
.venv/bin/python -m tests.test_standard
.venv/bin/python -m tests.test_bayes_exhaustive
.venv/bin/python -m tests.test_regression cpu
```

Each prints a summary and exits nonzero on failure. Pytest also collects `test_bayes_exhaustive.py`; it does not invoke the other module entry points. The regression runner accepts `--level 1` for opcode tests, `2` for instruction sequences, `3` for compiled C, or `4` for different machine sizes. Its default, `0`, runs all levels. Use `cuda` instead of `cpu` to include cases skipped by the CPU runner.

## Select a suite

Pass one or more paths to pytest:

```bash
.venv/bin/python -m pytest tests/test_python.py tests/test_score_options.py -q
```

| Area | Files under `tests/` |
|---|---|
| Scoring and Python translation | `test_score_options.py`, `test_python.py`, `test_decision.py` |
| Model helpers and package imports | `test_gemma.py`, `test_public_api.py` |
| Compiler | `test_comprehensive.py`, `test_compiler_ext.py`, `test_fuzz.py` |
| Arithmetic and vectors | `test_bayes.py`, `test_vector.py`, `test_swap.py` |
| Generated runtimes | `test_runtime.py`, `test_probabilistic.py` |
| Hardware host code | `test_icestick.py`, `test_icestick_transformer.py` |
| CUDA binary metadata | `test_prebuilt.py` |

The hardware host suites use offline peers. [RTL and physical-board tests](run-icestick.md) run separately; the differential RTL harness is `hardware/icestick/test_core.py`.

Increase the deterministic fuzz corpus with:

```bash
env STRANDEDCPU_FUZZ=200 .venv/bin/python -m pytest tests/test_fuzz.py -q
```

## Node and CUDA

The JavaScript comparison requires `node` on `PATH`:

```bash
.venv/bin/python -m pytest tests/test_js_interpreter.py -q
```

CUDA tests require an NVIDIA GPU supported by the installed PyTorch build. Some kernel-build cases also require nvcc:

```bash
.venv/bin/python -m pytest tests/test_cuda.py -q
```

Those suites skip when their runtime is unavailable. `test_prebuilt.py` needs no GPU; it checks the packaged binary's source hash. To rebuild that binary for all configured targets, use nvcc 12.8 or newer:

```bash
.venv/bin/python -m strandedcpu.cuda build-fatbin
```
