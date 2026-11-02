# StrandedCPU

StrandedCPU compiles small Python and C programs to a fixed instruction set, checks their arithmetic, and matches the result against numeric answer choices. Programs run on a CPU interpreter or a batched CUDA kernel. The eight-layer transformer can execute the same instructions through matrix operations.

This is a fork of [LOOM](https://github.com/mkturkcan/Loom). It adds exact fractions, checked C/Python execution, answer scoring, CUDA batching, and an extended 53-opcode ISA. [LOOM and this fork](docs/explanation/project-scope.md) describes the changes and credits the original work.

## Try it

From a checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

```python
from strandedcpu import score_python

answer = score_python(
    "def solution(): return 50 * 0.8",
    {},
    {"A": "$40", "B": "$41"},
    N=32,
    numbers="rational",
)

assert answer["status"] == "ok"
assert answer["match"] == {"A": True, "B": False}
print(answer["value"])  # 40
```

`numbers="rational"` uses fraction arithmetic. Float literals first pass through Python's float parser, so long literals can lose digits before execution. Arithmetic also has a finite range: an overflow returns an abstention reason instead of matching a wrapped result. The [Python tutorial](docs/tutorials/checked-python.md) walks through successful matches and abstentions.

The default backend is the CPU interpreter. `backend="auto"` tries CUDA and records `fallback_reason` if it uses the interpreter. Both support words up to 64 bits; CUDA binary fixed-point checks stop at 32 bits. The transformer's supported range ends at 16 bits.

## iCEstick

The [UART-loaded ISA core](docs/how-to/run-icestick.md) runs an 8-bit subset with 32 memory words and 128 instruction slots. It uses wrapping arithmetic. The separate [transformer experiment](docs/explanation/icestick-transformer.md) sends selected dot products to an FPGA MAC while the host stores the model and runs the remaining operations.

## Docs and tests

[Tutorials](docs/tutorials/README.md) · [How-to guides](docs/how-to/README.md) · [Reference](docs/reference/README.md) · [Explanation](docs/explanation/README.md)

Python tests live in `tests/`:

```bash
.venv/bin/python -m pytest
```

The [test guide](docs/how-to/run-tests.md) also covers standalone runners, Node.js comparisons, and GPU checks.

LOOM is by Mehmet Kerem Turkcan. See the [original paper](https://arxiv.org/abs/2604.08816), [citation](docs/explanation/project-scope.md#citation), [NOTICE](NOTICE), and [Apache-2.0 license](LICENSE).
