# iCEstick hardware

The SC8 image executes an 8-bit ISA profile with 32 memory words and 128 instruction slots. The host loads programs over UART and reads back memory, PC, carry, and the stop reason. Arithmetic wraps at 8 bits; checked scoring, exact fractions, and vector instructions are outside this profile.

The SCM1 image computes streamed dot products for the hybrid transformer backend. It multiplies signed 16-bit operands and accumulates into 32 bits. The host holds the weights and state and schedules the layers.

- [Build, load, and run the board](../../docs/how-to/run-icestick.md)
- [ISA profile, UART protocol, and board resources](../../docs/reference/icestick.md)
- [Transformer offload design and limits](../../docs/explanation/icestick-transformer.md)
- [Recorded board validation](validation.json)

`make` builds SC8; `make mac` builds SCM1. Keep flash backups in `backups/`, outside the disposable `build/` directory. Git ignores both directories.
