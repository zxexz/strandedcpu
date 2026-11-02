# Repository map

Production Python code is under `strandedcpu/`; Python tests are under `tests/`. Hardware sources and their simulation fixtures live together under `hardware/icestick/`.

## Execution and front ends

| Path | Purpose |
|---|---|
| [`strandedcpu/__init__.py`](../../strandedcpu/__init__.py) | Public program-execution and scoring entry points |
| [`strandedcpu/transformer.py`](../../strandedcpu/transformer.py) | Eight-layer transformer, configuration, opcode constants, state initialization, and decoding |
| [`strandedcpu/standard.py`](../../strandedcpu/standard.py) | Broadcast-bias transformer variant |
| [`strandedcpu/subleq.py`](../../strandedcpu/subleq.py) | Base transformer layers, bipolar encoding, and hard top-2 attention helpers |
| [`strandedcpu/interpreter.py`](../../strandedcpu/interpreter.py) | Python execution of the full ISA and checked arithmetic |
| [`strandedcpu/reference.py`](../../strandedcpu/reference.py) | Plain-Python extended arithmetic, carry, float-assist, and vector reference functions |
| [`strandedcpu/cuda.py`](../../strandedcpu/cuda.py) | Batched CUDA ISA kernel, compilation/loading paths, and packed programs |
| [`strandedcpu/kernel.fatbin`](../../strandedcpu/kernel.fatbin), [`strandedcpu/kernel.fatbin.json`](../../strandedcpu/kernel.fatbin.json) | Prebuilt CUDA code and source-hash/build metadata |
| [`strandedcpu/compiler.py`](../../strandedcpu/compiler.py) | C parser, code generation, compiler checks, and exact fraction runtime |
| [`strandedcpu/python.py`](../../strandedcpu/python.py) | Python-subset translation and scoring |
| [`strandedcpu/api.py`](../../strandedcpu/api.py) | Program execution, option parsing, checked scoring, and batching |
| [`strandedcpu/decision.py`](../../strandedcpu/decision.py) | Decision-model rows and fixed-order per-option features |
| [`strandedcpu/runtime.py`](../../strandedcpu/runtime.py) | C runtime-source and lookup-table generators |
| [`strandedcpu/host.py`](../../strandedcpu/host.py) | Mailbox streaming into interpreter or transformer state |
| [`strandedcpu/gemma.py`](../../strandedcpu/gemma.py) | Helpers for generating and checking language-model-written programs |
| [`strandedcpu/icestick.py`](../../strandedcpu/icestick.py) | Raw 8-bit FPGA program compiler, UART loader, and state verification |
| [`strandedcpu/icestick_transformer.py`](../../strandedcpu/icestick_transformer.py) | Hybrid transformer arithmetic dispatch with explicit host fallback |
| [`hardware/icestick/`](../../hardware/icestick/README.md) | Native iCE40 RTL, pin constraints, simulation, build targets, and board verification |
| [`javascript/interpreter.js`](../../javascript/interpreter.js) | Raw ISA interpreter used for cross-language agreement checks; no checked scoring interface |

## Tests

Paths in this table are under [`tests/`](../../tests/__init__.py). Commands and dependencies are in [Run the tests](../how-to/run-tests.md).

| Files | Coverage |
|---|---|
| `test_transformer.py`, `test_standard.py`, `test_swap.py`, `test_regression.py` | Base opcodes, architecture variants, and regressions |
| `test_comprehensive.py`, `test_compiler_ext.py`, `test_fuzz.py` | Compiled C and compiler semantics, including independent arithmetic evaluation |
| `test_bayes.py`, `test_bayes_exhaustive.py`, `test_vector.py` | Arithmetic extensions, exhaustive small-word checks, and vector datapaths |
| `test_runtime.py`, `test_probabilistic.py` | Generated runtimes and composed numerical/probabilistic programs |
| `test_score_options.py`, `test_python.py`, `test_decision.py` | Checked scoring, Python translation, and decision features |
| `test_gemma.py`, `test_public_api.py` | Model helper batch alignment, option labels, and public package imports |
| `test_icestick.py`, `test_icestick_transformer.py` | Hardware host protocols, constrained compilation, numeric dispatch, and hybrid graph behavior |
| `test_js_interpreter.py` | JavaScript/Python interpreter agreement; needs Node.js |
| `test_cuda.py`, `test_prebuilt.py` | CUDA/interpreter agreement and prebuilt-kernel freshness; GPU tests need CUDA |

[`hardware/icestick/test_core.py`](../../hardware/icestick/test_core.py) compares the RTL with the ISA interpreter. UART and integration benches are in `hardware/icestick/tests/`; the MAC bench is `hardware/icestick/mac_tb.v`. Physical checks use `verify_board.py` and `verify_transformer.py` in the same hardware directory.
