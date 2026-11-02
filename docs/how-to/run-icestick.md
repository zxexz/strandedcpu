# Run StrandedCPU on an iCEstick

The compact ISA image runs integer programs on a stock Lattice iCEstick, iCE40HX1K-TQ144. It has 32 data words and room for 128 instructions. Arithmetic wraps at eight bits; see the [hardware profile](../reference/icestick.md) for supported opcodes.

## Build

Install the package, plus Yosys, nextpnr-ice40, IceStorm (`icepack`, `iceprog`), and Icarus Verilog (`iverilog`, `vvp`). [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) bundles the FPGA tools; put its `bin` directory on `PATH`.

From the repository root:

```bash
make -C hardware/icestick test
.venv/bin/python hardware/icestick/test_core.py
make -C hardware/icestick
```

The first two commands run the UART/integration benches and differential ISA simulation. The build produces `hardware/icestick/build/top.bin`. Check `yosys.log` and `timing.json` under `hardware/icestick/build/` after RTL changes; the [recorded design](../../hardware/icestick/validation.json) uses 1,252 of 1,280 logic cells.

## Save the existing flash

A stock iCEstick boots from SPI flash. IceStorm's `-S` SRAM programming mode requires board modifications; use flash programming for an unmodified board.

Choose a backup filename that does not already exist:

```bash
mkdir -p hardware/icestick/backups
iceprog -d i:0x0403:0x6010 -R 4M hardware/icestick/backups/original-flash.bin
```

Keep the full 4 MiB backup. `make clean` leaves `backups/` intact.

## Program and run

FTDI channel A programs flash; channel B supplies UART. Replace `/dev/cu.usbserial-21101` with channel B's serial device on your machine.

```bash
iceprog -d i:0x0403:0x6010 hardware/icestick/build/top.bin
.venv/bin/python -m strandedcpu.icestick --port /dev/cu.usbserial-21101 --verify
```

Programming should finish with `VERIFY OK` and `cdone: high`. The default C demo returns `8` for `7 + 1`, with `status: halt` and `verified_against_interpreter: true`. Verification checks all memory words, PC, carry, and retired instruction count.

Upload another program over UART without reflashing:

```bash
.venv/bin/python -m strandedcpu.icestick --port /dev/cu.usbserial-21101 --source hardware/icestick/examples/multiply.c --input x=7 --verify
.venv/bin/python -m strandedcpu.icestick --port /dev/cu.usbserial-21101 --source hardware/icestick/examples/add.py --language python --input x=7 --verify
```

These return `22` and `8`. The compiler rejects noninteger source, unsupported opcodes, and programs exceeding the hardware limits.

Run the physical regression cases with:

```bash
.venv/bin/python hardware/icestick/verify_board.py --port /dev/cu.usbserial-21101 --output hardware/icestick/build/isa-on-device.json
```

For a UART timeout, check the channel, close other programs using the port, and confirm `top.bin` is loaded. An interrupted upload leaves the parser waiting for more bytes; reconnect the board before retrying the whole upload.

## Switch to transformer arithmetic

The separate `mac.bin` image evaluates selected transformer dot products. State, weights, attention selection, and remaining arithmetic stay on the host.

```bash
make -C hardware/icestick test-mac mac
iceprog -d i:0x0403:0x6010 hardware/icestick/build/mac.bin
.venv/bin/python -m strandedcpu.icestick_transformer --port /dev/cu.usbserial-21101 --max-offloaded-dots 4 --steps 2 --verify
```

The INC/HALT example should finish with memory word zero `6` and PC `0`. Its report gives FPGA dot counts and host fallbacks. A four-dot budget is spent in the first projection; later layers run on the host. See [transformer arithmetic](../explanation/icestick-transformer.md) for dispatch rules.

To check signed arithmetic, overflow recovery, an FFN row and ReLU, and every layer of two complete steps:

```bash
.venv/bin/python hardware/icestick/verify_transformer.py --port /dev/cu.usbserial-21101 --output hardware/icestick/build/transformer-validation.json
```

Reload `top.bin` to use the compact ISA interface again:

```bash
iceprog -d i:0x0403:0x6010 hardware/icestick/build/top.bin
```

## Restore the saved design

Use the backup created above:

```bash
iceprog -d i:0x0403:0x6010 hardware/icestick/backups/original-flash.bin
```
