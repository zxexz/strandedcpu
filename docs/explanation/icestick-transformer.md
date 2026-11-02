# Transformer arithmetic on iCEstick

The MAC image lets the iCEstick evaluate selected dot products from the transformer. The host keeps the weights and state, selects attention entries, and runs any arithmetic that is too wide or exceeds the chosen offload budget. This is different from the [ISA core](../reference/icestick.md), which decodes instructions on the FPGA.

The recorded INC/HALT run used the FPGA for 16 projection dots. Another 804,208 dots ran on the host. All 16 layer outputs across two instructions matched PyTorch. A separate check sent one FFN dot with ReLU to the board. Most of the transformer still runs on the CPU.

## Why the model stays on the host

An iCE40HX1K has 1,280 logic cells and 8 KiB of block RAM. With `Config(s=32, m=8, n=64, N=8)`, the transformer state is a 120-by-64 float32 tensor: 30,720 bytes before workspace. Its weight and bias tensors take another 3,888,064 bytes. The [Lattice board guide](https://www.latticesemi.com/view_document?document_id=50701) and [family datasheet](https://www.latticesemi.com/view_document?document_id=49312) describe the device.

[mac_core.v](../../hardware/icestick/mac_core.v) uses a 16-cycle shift/add multiplier and a signed 32-bit accumulator. [mac_top.v](../../hardware/icestick/mac_top.v) applies the optional ReLU to its result. Operands and counters live in registers. The routed image uses 698 logic cells, no RAM blocks, and no PLL. It runs from the board's 12 MHz oscillator; nextpnr estimated a 65.78 MHz maximum clock.

## Where the MAC enters the graph

Each layer computes:

```text
QX = Q X                  KX = K X                  VX = V X
S  = transpose(KX) QX
H  = VX A(S)
T  = X + sum_heads(H)
R  = ReLU(W1 T + b1)
Y  = T + W2 R + b2
```

[TransformerOffload](../../strandedcpu/icestick_transformer.py) follows these operations. It can send individual projection, score, value-mixing, or FFN dots to the board. `A(S)`, residual additions, and layer scheduling stay on the host. `max_offloaded_dots` limits the total number of hardware calls across steps, so a small budget can be exhausted in the first projection.

The host uses the same top-two rule as [subleq.py](../../strandedcpu/subleq.py): a unique winner gets weight 1; two scores less than 1 apart get 0.5 each. Dead sources contribute zero. Query-wise heads select per destination rather than per source.

## Which dots fit

Before sending a dot, the host finds power-of-two scales that represent both input vectors exactly as signed 16-bit integers. It adds the bias as another operand pair, checks the term count, and bounds the sum of absolute scaled products:

```text
sum(abs(a[i]) * abs(b[i])) <= min(2^31 - 1, 2^p)
p = 24 for float32, 53 for float64
```

The 32-bit bound prevents accumulator overflow. The floating-point bound keeps products and partial sums exactly representable in the source dtype, regardless of accumulation order. After receiving the integer result, the host restores the scale. ReLU runs after the bias and all products are included. The FPGA also detects intermediate overflow and returns an error.

Many model values fail these checks. Layers 1–7 use quarter increments, but some layer-2 biases reach 924,872. Layer 8 includes float32 approximations to `0.1` and `100.9`, which cannot be represented by the small integer scaling used here. Those dots stay on the host. Nonfinite values and exhausted budgets also cause host fallback; every reason is counted in `runner.stats`.

Wider arithmetic would need a different encoding, such as several integer limbs per coefficient. Reconstructing an exact sum and rounding once can differ from PyTorch's intermediate rounding, so extending the offload also requires a defined rounding rule.

## Running it

The [board guide](../how-to/run-icestick.md#switch-to-transformer-arithmetic) covers building and loading `mac.bin`. The image identifies itself as `SCM1`; the ISA image identifies itself as `SC8` followed by version byte 1. The [hardware reference](../reference/icestick.md) describes both protocols.

```bash
.venv/bin/python -m strandedcpu.icestick_transformer --port /dev/cu.usbserial-21101 --max-offloaded-dots 4 --steps 2 --verify
```

This runs INC and HALT through all eight layers. Memory word zero starts at 5 and should end at 6 with PC zero. `--verify` compares each step's float32 state with PyTorch at absolute tolerance `1e-5` and checks decoded memory and PC exactly.

For Python callers, `MacDevice.open(port)` opens the arithmetic transport. `TransformerOffload(board, max_offloaded_dots=4).step(computer, state)` returns the next tensor; `layer_outputs` holds its eight intermediate states. `offload_one_affine` runs the same dispatch for a single row and column, and `reset_stats()` resets the cumulative budget and counters.

## Board results

On 2 November 2026, [verify_transformer.py](../../hardware/icestick/verify_transformer.py) checked signed multiplication limits, zero terms, both overflow directions, and recovery on the next request. It then ran:

| Work | FPGA arithmetic | Result |
|---|---|---|
| L1 FFN row 1, column 0 | One three-term dot, including bias, followed by ReLU | 2, matching PyTorch |
| INC followed by HALT | 16 one-term projection dots; 804,208 host fallback dots | Memory word zero 6, PC 0; all 16 layer errors 0 |

The [saved summary](../../hardware/icestick/validation.json) records the configuration, timing estimates, bitstream hashes, and coverage. The full JSON can be regenerated with:

```bash
.venv/bin/python hardware/icestick/verify_transformer.py --port /dev/cu.usbserial-21101 --output hardware/icestick/build/transformer-validation.json
```

## UART cost

At 115,200 baud with 8N1 framing, the link carries 11,520 bytes per second. Each operand pair takes four bytes before request framing and replies, limiting the payload rate to 2,880 terms per second.

For this configuration, multiplying every nonzero Q/K/V/W coefficient across the state takes 1,118,592 terms. Dense attention scores add 344,064. Sending those pairs alone would take about 508 seconds, before biases, value mixing, framing, and return traffic. Skipping zero activations reduces that traffic. The small offload budget keeps the test short; it does not measure a whole-pass FPGA speedup.
