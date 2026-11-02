# iCEstick interfaces

Two images share the iCEstick's UART pins. `top.bin` executes the compact ISA; `mac.bin` evaluates transformer dot products. Each has a different firmware identifier. Neither is a scoring API backend value.

## Compact ISA profile

| Setting | Value |
|---|---|
| Word | Signed 8-bit two's complement; arithmetic wraps |
| Scratchpad offset `s` | 32 |
| Data memory `m` | 32 words; logical indices 0–31, column addresses 32–63 |
| Compiler address space `n` | 256 |
| Code | 1–128 triples of unsigned bytes `(a,b,c)` |
| Initial PC | 64 |
| Step limit | 1–65,535 retired instructions, including HALT |
| Named opcodes | 0–15, 17–21, 24–30, and JMPI 64 |
| SUBLEQ | Source columns 32–63 |
| Unsupported | FIND, LOGADD, XLAT, ARGMAX, float assists, and vectors |

Code and data occupy separate synchronous block RAMs. LOAD/STORE use the pointer's low five bits. Invalid direct data addresses and PC targets fault; the Python interpreter instead tolerates some invalid addresses.

`compile_program` rejects out-of-range literals, noninteger source, unsupported opcodes, and invalid host inputs. Runtime-check metadata is not uploaded, so the board does not check user arithmetic for overflow.

## Compact ISA host API

```python
from strandedcpu.icestick import IceStick, compile_program, verify

program = compile_program("int x; int main() { return x + 1; }", {"x": 7})
with IceStick.open("/dev/cu.usbserial-21101") as board:
    board.ping()
    board.load(program)
    result = board.run(max_steps=1000)
verify(program, result, max_steps=1000)
print(result.as_dict(program))
```

`compile_program(source, inputs=None, *, language="c")` also accepts `language="python"`. Inputs name uninitialized C scalar globals or Python entry parameters and must be signed 8-bit integers. `Program.payload()` validates words, instruction fields, count, and opcode support.

`Result` holds `fault`, `steps`, `pc`, `carry`, and 32 signed memory words. `as_dict(program)` adds variables, the selected result, `profile="raw-int8"`, and `checked=False`. `verify` requires HALT and compares memory, PC, carry, and instruction count with the raw interpreter.

The POSIX client uses standard-library serial I/O at 115,200 baud, 8N1, with a five-second default timeout. Replace the example port with FTDI channel B's device on the host.

### ISA wire format

Multibyte fields are little-endian. Send one request and read its reply before the next request.

| Command byte | Following request bytes | Reply |
|---|---|---|
| `P` (`0x50`) | None | `SC8` and version byte `0x01` |
| `L` (`0x4c`) | 32 memory bytes, instruction count, then three bytes per instruction | `K`, or `E` for an invalid count |
| `R` (`0x52`) | Unsigned 16-bit step limit | 38-byte result |
| `X` (`0x58`) | None | `K`; clears execution state but preserves code and data |

A result is `D`, fault byte, 16-bit step count, PC, carry, then 32 two's-complement memory bytes. Fault codes are 0 for HALT, 1 for unsupported opcode, 2 for invalid address, and 3 for step limit. Each run starts at PC 64 with cleared carry and count; memory retains the previous run's contents.

## SCM1 arithmetic image

Import from [`strandedcpu.icestick_transformer`](../../strandedcpu/icestick_transformer.py):

```text
MacDevice.open(port, timeout=5.0)
board.ping()
board.dot(pairs, relu=False)

TransformerOffload(board, max_offloaded_dots=None)
runner.offload_one_affine(coefficients, activations, bias=0.0, relu=False)
runner.step(computer, state)
runner.reset_stats()
```

Use `MacDevice` as a context manager to close the transport. `ping()` requires the four-byte `SCM1` reply and rejects the ISA image. `dot` accepts up to 65,535 signed 16-bit operand pairs, starts from zero, and returns a signed 32-bit sum. With `relu=True`, a negative completed sum becomes zero. Any intermediate accumulator overflow raises `OverflowError`, even if later terms or ReLU would hide it.

`TransformerOffload` evaluates the actual graph with the board's dots. It retains state, weights, attention selection, residual additions, and scheduling on the CPU. `step` takes a CPU float32 state matching the computer's shape and executes every layer. `offload_one_affine` takes equal-length CPU float32 or float64 vectors.

Dispatch requires exact power-of-two scaling of operands into signed 16-bit integers and a representable bias. The sum of absolute scaled products must fit signed 32 bits and be at most `2^24` for float32 or `2^53` for float64. Those bounds preserve products and accumulation in the source dtype. Rejected dots use PyTorch and increment a fallback reason.

`max_offloaded_dots` is a cumulative budget, not a per-layer budget. Zero is allowed and performs no board arithmetic. `reset_stats` clears the budget counter. `runner.stats.as_dict()` contains dot/term counts, hardware ReLU calls, zero dots, fallback counts by reason, and request/reply byte counts. `runner.layer_outputs` stores the most recent step's intermediate states. A small budget can be exhausted in the first projection.

### SCM1 wire format

| Command byte | Following request bytes | Reply |
|---|---|---|
| `P` (`0x50`) | None | ASCII `SCM1` |
| `D` (`0x44`) | Flags byte, unsigned 16-bit count, then pairs of signed 16-bit coefficient and activation | `A`, signed 32-bit result, overflow byte |

Flag bit 0 enables ReLU; other bits are reserved and the client sends zero. Zero terms return zero. An overflow byte of 1 invalidates the result. Unknown command bytes return `E`. Fields are little-endian; requests and replies are sequential.

Both protocols lack checksums and partial-packet timeouts. A truncated upload can leave the parser waiting for data; reconnect or reconfigure the board to reset it. Bytes received during execution or reply transmission are not queued.

## Pins and builds

| Signal | FPGA pin |
|---|---|
| 12 MHz clock input | 21 |
| UART TX to FTDI | 8 |
| UART RX from FTDI | 9 |
| LEDs 0–4 | 99, 98, 97, 96, 95 |

Pin assignments are in [`icestick.pcf`](../../hardware/icestick/icestick.pcf) and the [Lattice board guide](https://www.latticesemi.com/view_document?document_id=50701). The [HX1K datasheet](https://www.latticesemi.com/view_document?document_id=49312) specifies 1,280 logic cells and sixteen 4-Kibit RAM blocks.

| Make target | Output or action |
|---|---|
| Default | `build/top.bin`, compact ISA |
| `mac` | `build/mac.bin`, SCM1 arithmetic |
| `test` | UART, loader, and ISA integration simulations |
| `test-mac` | MAC and SCM1 UART simulations |
| `clean` | Remove the configured build directory |

Programming is a separate command; no Make target flashes the board. [Run the iCEstick](../how-to/run-icestick.md) covers setup, flash backup, programming, and validation.

The [2 November 2026 validation record](../../hardware/icestick/validation.json) identifies the tool versions and bitstream hashes. The routed ISA image used 1,252 logic cells and two RAM blocks; SCM1 used 698 cells and no RAM. Both met the 12 MHz routing constraint. Reported maximum clocks are tool estimates, not measured instruction throughput.
