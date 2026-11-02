# How the transformer executes instructions

The transformer stores instructions, data, and its program counter in one tensor. A pass through eight layers executes one instruction. Feed the output back in to execute the next one; PC zero means HALT. The weights depend on the machine configuration, so another program of the same shape uses the same weights.

[transformer.py](../../strandedcpu/transformer.py) builds the layers. [subleq.py](../../strandedcpu/subleq.py) implements their attention and feed-forward operations. The [first-program tutorial](../tutorials/first-program.md) steps through a two-instruction example.

## State layout

The state has shape `(cfg.d_model, cfg.n)`. Columns `0..s-1` are scratchpad, `s..s+m-1` are data memory, and instructions start at `s+m`. Rows hold instruction fields, word bits, the PC, position encodings, buffers, address tags, and flags.

A bit is `-1` for zero or `+1` for one. Words use two's complement. Column zero holds the PC and scalar carry. Vector carries live at their memory columns. Address tags let an indirect read select a word by index.

Raw instruction addresses are usually column indices: data word `i` is column `s+i`, and instruction `j` is column `s+m+j`. The compiler adds these offsets. LOAD, STORE, and JMPI have different pointer conventions, listed in the [ISA reference](../reference/instruction-set.md).

## The eight layers

| Layer | Work |
|---|---|
| L1 | Fetch the instruction at PC |
| L2 | Read operands and decode the opcode |
| L3 | Read indirect operands and correct scratchpad values |
| L4 | Subtract through a direct borrow chain; compute extended arithmetic |
| L5 | Write memory, including SWAP's second result |
| L6 | Compute the branch flag and PC increment |
| L7 | Select a branch destination or the next instruction |
| L8 | Correct the state for the next pass |

Opcode gates enable the feed-forward rows for the selected instruction. New operations add rows and attention heads to these layers without adding another layer.

## Attention

Hard attention selects the two highest scores. If their gap is below `1.0`, both get weight `0.5`; otherwise the winner gets weight `1.0`. Address heads arrange a tie between the control column and the addressed column. The feed-forward network rescales the copied bits.

The address heads use symmetric query and key projections. Vector broadcast needs a different direction: each destination chooses a source, allowing many columns to read the same source. Vector gather uses this query-wise selection too.

Scores accumulate in float64 because weighted comparisons can exceed float32's exact integer range. State and weight tensors normally remain float32, as do the threshold networks. The transformer's supported word width ends at 16 bits; the direct ISA implementations support wider words.

The optional softmax straight-through estimator changes the backward pass while retaining the hard forward result. The analytical weights can be constructed and executed without training.

## Arithmetic in the feed-forward networks

For integer `x` and `t`,

```text
ReLU(x - t + 1) - ReLU(x - t) = [x >= t]
```

Gating these threshold rows on the opcode gives comparisons, shift selection, carry bits, and LOGADD's correction table. A sum's low bits determine each output bit. MULSTEP uses that fact to add and shift in L2; DIVSTEP sends its conditional subtraction through L4.

MULSTEP and DIVSTEP work on two words at `b` and `b ^ 1`. L1 retargets a read head to the partner, and the second write head stores its result. ADD and ADC record carry; SUB records no borrow. The compiler combines these operations for multiword `fix` arithmetic.

ARGMAX uses attention to select a block's largest value, preferring the lowest address on a tie, then reads back the winner's index. LOGADD adds log-domain values by taking their maximum and looking up a correction for their difference.

## Page 2 and vectors

At `n >= 128`, the instruction fields can encode the second opcode page. JMPI supports returns from `noinline` functions. FALIGN and FNORM work on exponent/mantissa pairs. Vector operations act on aligned blocks of `VL = 2^t` words.

L4 broadcasts the instruction and vector mask to memory columns, then computes block membership and source addresses. L5 gathers the source words. The feed-forward rows in L5–L7 compute and store each element.

The source can be a corresponding block element, XOR partner, bit-reversed index, transposed index, or table entry. XOR-partner stages reduce a block in `log2(VL)` instructions. VBFLY performs a Walsh–Hadamard stage. VRAND hashes a seed and column address.

For example, categorical sampling uses VRAND for noise, VXLAT for a Gumbel lookup, VADD for log probabilities, and ARGMAX to select the result. [Compiler and runtime reference](../reference/compiler.md) lists the builtins.

## Broadcast biases

[standard.py](../../strandedcpu/standard.py) adds an indicator row to fold column-dependent biases into the weights. It uses broadcast bias vectors in the transformer layers.
