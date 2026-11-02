# Compile and run a C program

Use the environment from [the Python tutorial](checked-python.md#install). Run these examples from the repository root.

## Pass an input

Save `compiled_example.py`:

```python
from strandedcpu import run_function

source = """
int x;
int y;

int main() {
    y = x * x + 3;
    return y;
}
"""

out = run_function(source, {"x": 7})
print("y =", out["y"])
print("return =", out["__retval"])
```

```bash
.venv/bin/python compiled_example.py
```

```text
y = 52
return = 52
```

Declare inputs as globals without initializers. The host writes `x` before execution; an initializer such as `int x = 0;` would then overwrite it.

`run_function` returns named variables, the integer return value in `__retval`, and the instruction count in `__steps`. It uses the ISA interpreter by default. Arithmetic can wrap at the word width; the [scoring API](../how-to/score-options.md) adds arithmetic checks.

## Run the transformer too

Replace the file with a smaller program:

```python
from strandedcpu import run_function

source = "int x; int main() { return x + 1; }"

for backend in ("interpreter", "transformer"):
    out = run_function(
        source, {"x": 5}, N=8, s=32, m=32,
        backend=backend, trace=True,
    )
    print(backend, out["__retval"], out["__steps"])
```

Both paths should print:

```text
interpreter 6 4
transformer 6 4
```

The transformer executes the compiled instructions through its matrix operations. Four instructions are needed because the compiler also moves values into temporary and return slots. `out["__trace"]` records the PC, opcode, operands, and memory writes for each instruction.

Change `{"x": 5}` to `{"x": 9}` and run again. Both results should be `10`.

The [next tutorial](first-program.md) builds a transformer state directly. See the [C subset](../reference/compiler.md#c-subset) for language support and the [API](../reference/api.md#strandedcpurun_function) for return fields.
