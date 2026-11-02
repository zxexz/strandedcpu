# Arithmetic and backends

There are two choices to make: how numbers are stored, and what executes the instructions. A rational program can run on the CPU interpreter or CUDA ISA kernel. Running the transformer instead changes the computation used to execute each instruction, not the program's number format.

## Word arithmetic and `fix`

Memory words contain `N` bits. Raw ISA operations wrap at that width. Checked scoring uses compiler metadata to detect overflow and domain errors before executing the affected user operation. `run_function` runs without those checks.

A `fix` value takes two words. In `numbers="binary"` mode it stores a signed integer part and `N` fractional bits: Q8.8 at `N=8`, Q16.16 at `N=16`. Values such as `0.8` need rounding. A result near a decimal option can therefore fail an exact comparison.

With `numbers="rational"`, the words hold a reduced numerator and positive denominator. The compiler converts each parsed float literal to a fraction, so `0.1 + 0.2 == 0.3`. Both source parsers first read these literals as Python floats, however: `0.10000000000000001` becomes `1/10`, losing the final digit before fraction arithmetic begins. Each operation then uses integer arithmetic and reduction. Fractions cost more instructions, and their intermediate numerators and denominators can overflow even when the final value would be small. Wider words give them more room but make the bit-serial arithmetic slower.

Python division rules survive translation: `//` floors and `%` has the divisor's sign. `round(x)` uses ties to even; `round(x, n)` uses ties away from zero. In rational mode, `math.sqrt` and fractional powers require an exact root. An irrational result abstains as `inexact_sqrt`; `math.isqrt` returns the floor integer root. The [compiler reference](../reference/compiler.md) lists the supported operations.

## Matching and abstention

A scoring call succeeds when its result equals exactly one parsed option. It abstains on a compile or arithmetic fault, invalid input, a step limit, an unparseable option, or zero or multiple matches. Option keys are labels; renaming or reordering them does not change the arithmetic.

Rounding and failure are separate. Integer division with a remainder and rounded binary `fix` operations increment `inexact_ops` without necessarily causing abstention. `rounding_slack` permits a nearby match only when `fix_rounding > 0`; integer division alone does not enable it. Two options within the chosen slack still cause `multi_match`.

The result hashes identify the supplied program and repository state. They do not check the translation from a word problem to that program. Compiler tests also need an independent evaluator: two backends can agree on the same wrongly compiled instructions. The [fuzz tests](../../tests/test_fuzz.py) include that comparison.

## Execution paths

The interpreter executes word operations in Python. The CUDA ISA kernel gives each program a GPU thread and executes those operations directly. The transformer instead computes eight layers of attention and feed-forward networks for each instruction. Kernel timings and transformer timings measure different work.

The interpreter and CUDA ISA support words through 64 bits. CUDA binary fixed-point checks need intermediate products of `4N` bits, so binary-mode scoring above 32 bits uses the interpreter, even for integer-only programs. Rational mode can use CUDA through 64 bits. `backend="auto"` records a `fallback_reason` when it falls back.

The transformer accumulates attention scores in float64 but keeps its threshold networks in float32. Use `N <= 16`. The CPU transformer path accepts larger widths despite that precision limit; `cuda-transformer` rejects them.

The scoring backends `transformer` and `cuda-transformer` run both direct ISA execution and transformer execution, then compare the named variables in compiler metadata. They skip `__retval` and other machine state. A return-only C program with no named variables therefore has no values in that comparison. For direct transformer execution, use `run_function(..., backend="transformer")`. See the [API reference](../reference/api.md#backend-values) for the parameters.

The [iCEstick](../reference/icestick.md) has separate host interfaces: its ISA image wraps 8-bit words, and its MAC image evaluates selected transformer dots. Neither is a checked scoring backend.

## Compiler and transformer fixes

L3's correction interval is `0.25`, which is exactly representable in float32; the older `0.1` interval allowed drift in 16-bit subtraction. JNZ's gate weight grows with `N` so a one-bit opcode mismatch cannot activate it for wider operands. Signed comparisons use MAX instead of testing the sign of `a - b`, since that difference can overflow.

The compiler saves a live temporary when the next expression would overwrite it. This matters for nested calls and assignments such as `a[i] = a[j]`. LOAD/STORE also impose an indirect-address window: a variable-indexed array must fit in the first `s` memory words. The [compiler reference](../reference/compiler.md) covers that allocation limit.
