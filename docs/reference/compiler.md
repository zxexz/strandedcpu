# Compiler and runtime

The [C compiler](../../strandedcpu/compiler.py) emits ISA triples. The [Python front end](../../strandedcpu/python.py) translates to C before compilation.

## `compile_c`

```text
compile_c(source, s=32, m=64, n=1024, N=8,
          checked=False, numbers="binary")
```

Returns `(cfg, memory, commands, meta)`: configuration, initial memory words, `(a,b,c)` instructions, and metadata. For transformer execution, choose a power-of-two `n` with `s + m + len(commands) <= n`. Compilation does not check that capacity; `init_state` truncates excess instructions. Page 2 requires `n >= 128`.

`checked=True` rejects out-of-range literals with `ExecutionFault`. Arithmetic checks are always recorded in `meta["checks"]`; pass them to `ISAInterpreter(..., checks=meta["checks"])` to enforce them. [Option scoring](api.md#strandedcpuscore_options) does both.

| Metadata key | Contents |
|---|---|
| `variables`, `sizes` | Named variable memory offsets and allocated sizes |
| `fix_variables` | `fix` names and the offset of their first word |
| `constants` | Constant allocation map |
| `num_instructions` | Instruction count |
| `retval_addr`, `output_addr` | Return and output memory offsets |
| `labels`, `source_map` | Resolved labels and `(instruction_index, source_line)` mappings |
| `checks` | Checks indexed by instruction index |
| `fix_literal_rounding` | Count of binary literals that required rounding |
| `numbers` | Selected number mode |

Memory addresses in metadata are offsets relative to memory; emitted instruction operands are column addresses, offset by `s`.

## C subset

| Area | Supported |
|---|---|
| Types | Signed `N`-bit `int`; two-word `fix`; `void` function declarations |
| Expressions | `+ - * / %`, bitwise `& \| ^ ~`, unary `-`, shifts `<< >>`, comparisons `== != < > <= >=`, logical `&& \|\| !`, casts `(int)` and `(fix)` |
| Statements | Declarations, assignment, `+=`, `-=`, `if/else`, `while`, `for`, `break`, `continue`, `return` |
| Arrays | Constant-length integer arrays, constant initializers, constant or variable indexing |
| Functions | Inline by default; `noinline` compiles one body and returns through JMPI |

The subset has no pointers, structs, dynamic allocation, recursion or `fix` arrays. Functions do not have ordinary C lexical scopes; generated runtime identifiers use prefixes to avoid collisions. `printf(x)` and `output(x)` write their first argument to the designated output slot rather than formatting text.

Integer `/` truncates toward zero; `%` follows the dividend's sign. Comparisons remain valid when the difference between operands would overflow. Variable shifts emit SHIFT; multiplication and division emit MULSTEP/DIVSTEP sequences.

Variable array indexing uses LOAD/STORE, which can reach only the first `s` memory words. The compiler rejects an array indexed by a variable if its allocation extends beyond that window. Declare such arrays earlier or increase `s`. Temporaries, alignment, and runtime code also consume memory and instruction space.

`noinline` requires page 2 and supports calls through JMP/JMPI. These functions are not reentrant. `while (1)` emits no condition test.

## Number modes

| Mode | Representation and behavior |
|---|---|
| `numbers="binary"` | Default. A `fix` is a signed high word and unsigned low word: `hi + u(lo)/2^N`. Q8.8 at `N=8`, Q16.16 at `N=16`. Arithmetic can round at the fractional-word boundary. |
| `numbers="rational"` | A `fix` is a reduced numerator/positive-denominator pair. Arithmetic calls the checked fraction runtime. |

Both modes support literals, casts, `fix` addition/subtraction/multiplication/division and comparisons. C's `(int)` cast of `fix` rounds toward negative infinity. The Python front end implements its own `int()` truncation semantics.

In both front ends, float literals first pass through Python `float`; rational mode converts that parsed value with `Fraction(repr(value))`. Thus `0.10000000000000001` becomes `1/10`, even at `N=64`, where the fraction written in the source would fit. Integer literals do not pass through `float`.

Arithmetic is exact on the resulting fractions, so `0.1 + 0.2 == 0.3`. Overflow includes intermediate calculations; a small final fraction can still exceed the word range during computation.

## C builtins

| Builtin | Behavior |
|---|---|
| `abs(x)`, `min(a,b)`, `max(a,b)`, `mul(a,b)` | Integer absolute value, minimum, maximum and multiplication |
| `swap(a,b)` | Exchange operands with SWAP |
| `logadd(a,b)` | Log-domain addition in LOGADD units |
| `xlat(table,i)` | OR-addressed table lookup |
| `lfsr(state,taps)`, `clz(x)` | One LFSR step; leading zero count |
| `argmax(array)` / `argmax(array,k)` | Largest element's index over the array or its first `2^k` elements; first index wins ties |
| `umulhi(a,b)` | High word of the unsigned two-word product |
| `qdiv(a,b)` | Unsigned `(a << N) / b`, intended for `a < b` |
| `mulmod(a,b,p)` | Unsigned product modulo `p`, using a two-word product |
| `falign(f,e)`, `fnorm(f)` | Align/normalize an exponent-mantissa `int f[2]`; page 2 required |
| `diff(expr,x)` | Compile-time forward derivative of a `fix` expression with respect to scalar `x` |
| `fsqrt(x)`, `froot(x,k)` | Exact rational square/k-th root; irrational roots stop with `inexact_sqrt` |
| `isqrt(n)` | Floor integer square root; negative input is a runtime error |
| `fpowi(x,n)`, `ipowi(x,n)` | Runtime integer powers of a `fix` or integer base |

Roots and runtime powers require rational mode; their runtime modules are included on demand. Runtime exponents are limited to `abs(n) <= 4096`; exceeding the limit reports `step_limit`. Integer bases require nonnegative runtime exponents.

`diff` emits straight-line code. It supports arithmetic expressions and calls to single-return functions; general control flow in a differentiated function is unsupported. Arrays used by XLAT, ARGMAX and vector builtins are allocated with alignment for their size. `argmax(array)` and vector arrays require power-of-two lengths.

### Vector builtins

The compiler sets VL from the first array's size. Binary array operations require equal lengths; vector builtins require the vector unit (`n >= 128`).

| Arguments | Builtins |
|---|---|
| `(destination, source_array)` | `vmov`, `vadd`, `vsub`, `vmax`, `vlogadd`, `vcadd`, `vrev`, `vtrans` |
| `(destination, table)` | `vxlat` |
| `(destination, scalar)` | `vfill`, `vshift`, `vrand`, `vpermx`, `vaddx`, `vbfly`, `vmaxx`, `vlogaddx` |
| `(array)` | `vsum`, `vmaxall`, `vlogsum`, `wht` |

The reduction builtins mutate the array through XOR-partner stages. `vsum`, `vmaxall` and `vlogsum` leave the result in every element and return the first element. `wht` performs a Walsh–Hadamard transform in place. See the [instruction set](instruction-set.md#vector-opcodes) for element semantics.

## Python subset

The front end accepts functions, module constants, `import math`, and `from math import ...`. The entry is `solution`, the sole function, or the name supplied by `entry=`. Parameters must be plain positional parameters with no defaults or variadic arguments. The entry parameter name `result` is reserved.

| Area | Supported |
|---|---|
| Values and operators | Integer/float literals; `+ - * / // %`, unary `-`, comparisons including chains, `and`, `or`, `not`, conditional expressions, powers |
| Numeric functions | `abs`, `min`/`max` with multiple arguments, `int`, `float`, `round`, `math.floor`, `math.ceil`, `math.sqrt`, `math.isqrt`, `divmod` assigned to two names |
| Control flow | `if/elif/else`, `while`, `for` over `range` with constant step, `break`, `continue`, early `return` |
| Assignment | Names and supported list elements; augmented assignment to names; chained and tuple assignment |
| Sequences | Constant-length list literals, indexed reads/writes including negative indexes, `len`, `sum`, `min`, `max`, iteration over lists |
| Iteration helpers | `sum(range(...))`, `sum(expr for v in range(...) if cond)` |
| Output | `print` calls are ignored |

Python `/` produces `fix`. `//` floors; `%` has the divisor's sign, including float operands. `and` and `or` short-circuit and return operands. A variable becomes `fix` if any assignment requires it. Rational mode computes with fractions of the parsed values; [number modes](#number-modes) describes the float-literal parsing limit.

`round(x)` uses ties to even. `round(x,n)` requires a constant `n` in 0–6 and rounds ties away from zero: `round(0.125,2)` is `0.13`. In binary mode, decimal rounding can first lose precision in the binary `fix` value.

A constant exponent must reduce to `p/q` with `abs(p) <= 64` and `q <= 64`. Fractional powers require rational mode and an exact rational root; otherwise they abstain as `inexact_sqrt`. A negative root base produces a domain error. Runtime integer exponents use the rational power runtime and its limits above. `math.sqrt` requires an exact rational result; `math.isqrt(1000)` returns 31.

Lists use one variable per element. Aliasing, `append`, list parameters, and loops that both rebind and mutate the iterated list are refused. Unsupported syntax raises `PythonSubsetError` in translation or returns `compile_error` during scoring.

## Runtime generators

[`strandedcpu.runtime`](../../strandedcpu/runtime.py) generates C source to prepend to a program. Tables depend on `N` and the generator's scale arguments.

| Generator | Generated operations |
|---|---|
| `cordic(N)` | Sin/cos, atan, exp and ln in scaled integer words |
| `log_domain(N, frac_bits=2, table_bits=3)` | Linear/log2 conversions compatible with LOGADD |
| `newton(N)` | Reciprocal and square root iterations |
| `integer_ops(N)` | Popcount, gcd, saturating add and integer square root |
| `difference_engine(name, first_values, count)` | Polynomial tabulation by forward differences |
| `trapezoid(...)`, `simpson(...)` | Integration over power-of-two sample grids |
| `xorshift(N)`, `xorshift_seed(state,N)` | Multiword xorshift generator and seed source; see source for supported word widths |
| `gaussian_table(...)`, `gumbel_table(...)` | Inverse-distribution lookup tables |
| `qmul(N,F,name="rt_qmul")`, `fft(N,F,log2n)` | Scaled multiplication and complex fixed-point FFT |
| `ntt(n,p,root)` | Number-theoretic transform |
| `lns(N,frac_bits=2)`, `bfp(name,size)` | Logarithmic and block floating-point formats |
| `stochastic_rounding(N)`, `flt(N)` | Stochastic rounding and microcoded float |
| `mailbox(size)` | `mb`, `mb_len`, `mb_req` and `mb_wait()` for host streaming |

Read each generator's docstring for scales and input domains; CORDIC, for example, uses `N-3` fractional bits. The [host API](api.md#host-mailbox) supplies mailbox batches between steps.
