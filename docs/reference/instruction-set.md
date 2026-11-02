# Instruction set

There are 53 named opcodes: 0–31 and 21 page-2 operations, plus SUBLEQ mode. Constants live in [`transformer.py`](../../strandedcpu/transformer.py); direct semantics are in [`interpreter.py`](../../strandedcpu/interpreter.py) and [`reference.py`](../../strandedcpu/reference.py).

## State and addressing

The transformer state has shape `(cfg.d_model, cfg.n)`. Values use `N`-bit two's complement, encoded with bipolar bits (`-1` for 0, `+1` for 1).

| Columns | Contents |
|---|---|
| `0 .. s-1` | Scratchpad |
| `s .. s+m-1` | Memory words |
| `s+m .. n-1` | Instructions |

Instructions are triples `(a, b, c)`. `a` selects an opcode except in SUBLEQ mode; `b` and `c` are column addresses. Memory word `i` has address `s+i`. The initial PC is `s+m`; halted PC is 0. An ordinary instruction advances the PC by one unless it branches.

In the tables, `M[x]` means memory at column `x`; `u(v)` means the unsigned interpretation of an `N`-bit word. Writes wrap to `N` bits. LOAD and STORE pointers address `P(v) = s + (u(v) mod s)`, so indirect access reaches the first `s` memory words. Jump targets for JMP/JZ/JNZ/CMP are column addresses. JMPI instead takes an instruction index from memory.

## Base opcodes

| Value | Constant | Operation |
|---|---|---|
| 0 | `OP_HALT` | Set PC to 0 |
| 1 | `OP_MOV` | `M[b] = M[c]` |
| 2 | `OP_ADD` | `M[b] += M[c]`; set unsigned carry out |
| 3 | `OP_JMP` | `PC = c` |
| 4 | `OP_JZ` | If `M[b] == 0`, `PC = c` |
| 5 | `OP_JNZ` | If `M[b] != 0`, `PC = c` |
| 6 | `OP_INC` | `M[b] += 1` |
| 7 | `OP_DEC` | `M[b] -= 1` |
| 8 | `OP_SHL` | `M[b] <<= 1` |
| 9 | `OP_SHR` | Arithmetic `M[b] >>= 1` |
| 10 | `OP_CMP` | If signed `M[b] < 0`, `PC = c` |
| 11 | `OP_LOAD` | `M[b] = M[P(M[c])]` |
| 12 | `OP_AND` | `M[b] &= M[c]` |
| 13 | `OP_OR` | `M[b] \|= M[c]` |
| 14 | `OP_XOR` | `M[b] ^= M[c]` |
| 15 | `OP_SUB` | `M[b] -= M[c]`; carry becomes unsigned “no borrow” |
| 16 | `OP_FIND` | Store the lowest memory offset whose value equals `M[c]`; 0 if absent |
| 17 | `OP_SWAP` | Exchange `M[b]` and `M[c]` |
| 18 | `OP_CMOV` | If original `M[b] < 0`, `M[b] = M[c]` |
| 19 | `OP_MULACC` | `M[b] = (M[b] << 1) + (M[c] if original M[b] < 0 else 0)` |
| 20 | `OP_STORE` | `M[P(M[c])] = M[b]` |

## Arithmetic and lookup extensions

| Value | Constant | Operation |
|---|---|---|
| 21 | `OP_MAX` | Signed `M[b] = max(M[b], M[c])` |
| 22 | `OP_LOGADD` | `M[b] = max(M[b], M[c]) + T[abs(M[b]-M[c])]`, with zero correction outside the table |
| 23 | `OP_XLAT` | `M[b] = M[c \| low_bits(u(M[b]))]`; low bits are limited to `min(log2(s), N)` |
| 24 | `OP_SHIFT` | Shift `M[b]` left by `M[c]`; negative counts give arithmetic right shifts |
| 25 | `OP_ADC` | `M[b] += M[c] + carry`; set new unsigned carry out |
| 26 | `OP_MULSTEP` | One unsigned shift-add multiplication step on `(M[b], M[b^1])` using `M[c]` |
| 27 | `OP_DIVSTEP` | One unsigned restoring-division step on `(remainder, quotient) = (M[b], M[b^1])` using divisor `M[c]` |
| 28 | `OP_LFSR` | Logical right shift of `M[b]`, XOR with `M[c]` when the original low bit was 1 |
| 29 | `OP_CLZ` | Leading zero count of `u(M[c])`; zero gives `N` |
| 30 | `OP_ABS` | `M[b] = abs(M[c])`, with ordinary word wrapping |
| 31 | `OP_ARGMAX` | Index of the largest word in the `2^M[b]`-word block at `c`; lowest index wins ties |

`LOGADD` encodes log2 values in units of `2^-log_frac_bits`, with `log_frac_bits=2` by default. `logadd_table()` generates its correction table. XLAT combines addresses with bitwise OR, so a table must be aligned to its size and its index must fit that table.

The scalar carry starts at zero and persists between instructions. ADD and ADC set carry out; SUB sets 1 when unsigned subtraction needs no borrow. Other scalar instructions preserve it.

For multiplication, initialize the pair to `(0, multiplier)` and perform `N` MULSTEPs with the multiplicand to obtain a two-word product. For division, initialize `(0, dividend)` and perform `N` DIVSTEPs to obtain `(remainder, quotient)`. The partner address is the column address `b ^ 1`.

## Page 2

The transformer requires `n >= 128` to represent page-2 opcodes. `Config(vector_unit=False)` disables the vector unit while retaining other page-2 operations.

| Value | Constant | Operation |
|---|---|---|
| 64 | `OP_JMPI` | `PC = s + m + u(M[c])` |
| 65 | `OP_FALIGN` | For exponent/mantissa pair at `(b, b^1)`, align to `M[c]` if it is larger |
| 66 | `OP_FNORM` | Normalize that pair until the mantissa's top two bits differ; adjust exponent |
| 67 | `OP_VSETVL` | Set vector length to `VL = 2^M[b]` |

FALIGN arithmetic-shifts the mantissa right by at most `N-1` bits. FNORM encodes a zero mantissa with the most negative exponent. The float representation used by compiler builtins is `mantissa * 2^(exponent - (N-2))`.

### Vector opcodes

Let `B` and `C` be the `VL`-word aligned blocks containing `b` and `c`. Vector operations read all operands before writing any elements. `x[i]` below denotes the original element in `B`. The scalar operand is `M[c]`.

| Value | Constant | Operation per element |
|---|---|---|
| 68 | `OP_VMOV` | `B[i] = C[i]` |
| 69 | `OP_VADD` | `B[i] += C[i]`; set per-column carry |
| 70 | `OP_VSUB` | `B[i] -= C[i]`; set per-column no-borrow flag |
| 71 | `OP_VMAX` | `B[i] = max(B[i], C[i])` |
| 72 | `OP_VLOGADD` | Apply scalar LOGADD to `B[i]` and `C[i]` |
| 73 | `OP_VCADD` | Add the carry stored for `C[i]` into `B[i]`; update destination carry |
| 74 | `OP_VFILL` | `B[i] = M[c]` |
| 75 | `OP_VSHIFT` | Apply scalar SHIFT to `B[i]` with count `M[c]` |
| 76 | `OP_VXLAT` | Apply XLAT with base `c` and index `B[i]` |
| 77 | `OP_VREV` | `B[i] = C[bitreverse(i)]` over `log2(VL)` bits |
| 78 | `OP_VTRANS` | Transpose a square `2^(t/2)` block when `t=log2(VL)` is even; otherwise copy |
| 79 | `OP_VRAND` | Hash the seed `M[c]` and the destination memory offset |
| 84 | `OP_VPERMX` | `B[i] = x[i ^ mask]` |
| 85 | `OP_VADDX` | `B[i] = x[i] + x[i ^ mask]`; set per-column carry |
| 86 | `OP_VBFLY` | Walsh–Hadamard sum/difference stage, using the XOR partner |
| 87 | `OP_VMAXX` | `B[i] = max(x[i], x[i ^ mask])` |
| 88 | `OP_VLOGADDX` | LOGADD of `x[i]` and `x[i ^ mask]` |

For partner operations, `mask = u(M[c]) & (VL-1)`. A Walsh–Hadamard stage uses a power-of-two mask: columns with that bit clear receive the sum, and columns with it set receive the partner minus their own value. Vector carries persist separately from the scalar carry.

## SUBLEQ mode

For the transformer's SUBLEQ page, `a` in 32–63 is a source memory column rather than a named opcode:

```text
M[b] = M[b] - M[a]
if M[b] <= 0: PC = c
```

The interpreter treats any `a >= 32` that is absent from `PAGE2_OPCODES` as SUBLEQ. For transformer programs, use the documented 32–63 range; unassigned page-2 patterns are not additional named instructions.
