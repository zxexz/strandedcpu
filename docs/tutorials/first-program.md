# Run a transformer program

This program increments one memory word through the analytical transformer's eight layers.

Install the package from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

## Build the state

Save `first_program.py`:

```python
import torch

from strandedcpu.transformer import (
    Config, Computer, init_state, read_memory, get_pc,
    OP_INC, OP_HALT,
)

cfg = Config(s=32, m=8, n=64, N=8)
computer = Computer(cfg)

memory = [5] + [0] * 7
program = [
    (OP_INC, cfg.s, 0),
    (OP_HALT, 0, 0),
]
state = init_state(cfg, memory, program)

with torch.no_grad():
    for steps in range(1, 11):
        state = computer.step(state)
        if get_pc(state, cfg) == 0:
            break
    else:
        raise RuntimeError("program did not halt within 10 steps")

print("mem[0] =", read_memory(state, cfg)[0])
print("steps =", steps)
```

The configuration allocates 32 scratchpad columns and 8 memory columns within 64 total columns. Each memory word has 8 bits; instructions follow the memory columns.

Instructions are triples `(opcode, b, c)` with column addresses as operands. Memory word zero is at column `cfg.s`, so `(OP_INC, cfg.s, 0)` increments it. `init_state` packs these instructions, the initial memory, and the PC into a tensor.

## Execute it

```bash
.venv/bin/python first_program.py
```

```text
mem[0] = 6
steps = 2
```

Each `computer.step` passes through all eight layers and executes one instruction. INC changes `5` to `6`; HALT sets the PC to zero. `torch.no_grad()` avoids recording gradients during execution.

## Add an instruction

Insert another INC before HALT:

```python
program = [
    (OP_INC, cfg.s, 0),
    (OP_INC, cfg.s, 0),
    (OP_HALT, 0, 0),
]
```

Run the file again. Memory word zero should be `7` after three steps. The weights stay the same; the new instruction lives in the state tensor.

The [ISA reference](../reference/instruction-set.md) lists the other instructions. [Architecture](../explanation/architecture.md) explains the layers.
