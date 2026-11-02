# LOOM and this fork

StrandedCPU started from [LOOM commit 7148328](https://github.com/mkturkcan/Loom/commit/7148328). LOOM, by Mehmet Kerem Turkcan, supplied the 21-opcode machine, C compiler, and eight-layer transformer with analytically derived weights. Programs live in the state tensor; loading another program leaves the weights unchanged.

This fork extends the ISA to 53 named opcodes, with carry, multiword arithmetic, indirect calls, and vector operations. It adds a Python front end, binary fixed point and exact fractions, arithmetic checks, and numeric answer matching. The CPU interpreter and CUDA kernel execute the ISA directly. The transformer remains available for running the corresponding matrix operations and comparing states.

The compiler changes include signed comparisons that survive subtraction overflow, preservation of temporary values across sibling expressions and calls, and early returns from inlined functions. The transformer changes include word-size-dependent gates, float64 attention scores, and query-wise vector broadcast and gather. The [arithmetic](arithmetic-and-backends.md) and [architecture](architecture.md) pages explain these choices.

`score_options` and `score_python` check a program's arithmetic and look for a unique matching choice. They cannot tell whether the program represents the question correctly. Checked execution has finite word limits, and raw ISA execution wraps; see the [API reference](../reference/api.md) before choosing a backend.

## Retained and removed implementations

The eight-layer transformer and its broadcast-bias variant are maintained. The seven-layer merged variant was removed because it drops attention needed by ARGMAX and vector instructions. Old `loom_*` modules and `extended_isa_v4*` aliases were replaced by the `strandedcpu` package; the [migration guide](../how-to/migrate-from-loom.md) lists the new names.

The inherited browser site, ONNX exports, games, and Alveo prototype were removed. The [upstream repository](https://github.com/mkturkcan/Loom) retains that work. The [JavaScript interpreter](../../javascript/interpreter.js) stays as a second implementation for differential tests.

The [iCEstick RTL](../../hardware/icestick/README.md) was written for this fork. It has two images: a small wrapping integer ISA core and a MAC used by the [host-controlled transformer experiment](icestick-transformer.md). Neither implements the checked scoring API on the board.

## Citation

This citation is for the original LOOM architecture:

```bibtex
@misc{turkcan2026loomscalableanalyticalneural,
  title={Loom: A Scalable Analytical Neural Computer Architecture},
  author={Mehmet Kerem Turkcan},
  year={2026},
  eprint={2604.08816},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.08816}
}
```

LOOM's [paper](https://arxiv.org/abs/2604.08816), [browser demos](https://mkturkcan.github.io/Loom/demos/), and [model downloads](https://huggingface.co/mehmetkeremturkcan/Loom) describe the upstream machine. Attribution is recorded in [NOTICE](../../NOTICE); the code uses the [Apache-2.0 license](../../LICENSE).
