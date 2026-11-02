"""Standalone transformer regression runner.

Levels 1-4 cover individual opcodes, instruction sequences, compiled C, and
configuration sizes. Use --level 0 for all levels.

Run: python -m tests.test_regression [cpu|cuda] [--level 0|1|2|3|4] [--verbose]
"""

import sys
import time
import torch

from strandedcpu.transformer import (
    Computer, Config,
    init_state, read_memory, get_pc,
    OP_HALT, OP_MOV, OP_ADD, OP_SUB, OP_INC, OP_DEC,
    OP_AND, OP_OR, OP_XOR, OP_JMP, OP_JZ, OP_JNZ,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_FIND,
    OP_SWAP, OP_CMOV, OP_MULACC,
)
from strandedcpu.compiler import compile_and_run


# Configuration

DEFAULT_CFG = dict(s=32, m=64, n=1024, N=8)
SMALL_CFG = dict(s=32, m=8, n=64, N=8)
LARGE_CFG = dict(s=32, m=64, n=2048, N=8)


# Helpers

class TestResult:
    __test__ = False  # Result accumulator for the standalone regression runner.

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.t0 = time.monotonic()

    def ok(self, name, detail=""):
        self.passed += 1
        if detail:
            print(f"  PASS {name:40s}  {detail}")
        else:
            print(f"  PASS {name}")

    def fail(self, name, detail=""):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"  FAIL {name:40s}  {detail}")

    def err(self, name, exc):
        self.failed += 1
        self.errors.append((name, str(exc)))
        print(f"  ERR  {name:40s}  {exc}")

    def summary(self):
        elapsed = time.monotonic() - self.t0
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"{self.passed}/{total} passed in {elapsed:.1f}s")
        if self.errors:
            print(f"\nFailures:")
            for name, detail in self.errors:
                print(f"  {name}: {detail}")
        return self.failed == 0


def run_isa_test(name, cfg_kwargs, memory, commands, expected_mem, max_steps=50):
    """Run ISA-level test: init state, step until halt, check memory."""
    cfg = Config(**cfg_kwargs)
    ext = Computer(cfg)
    X = init_state(cfg, memory, commands)

    with torch.no_grad():
        for step in range(max_steps):
            pc = get_pc(X, cfg)
            if pc == 0:
                break
            X = ext.step(X)

    result = read_memory(X, cfg)
    for idx, expected_val in expected_mem.items():
        if result[idx] != expected_val:
            return False, f"mem[{idx}]: got {result[idx]}, expected {expected_val}"
    return True, ""


def run_c_test(name, source, expected, cfg_kwargs=None, device='cpu',
               max_steps=50000, max_seconds=90.0):
    """Compile C source and run on neural computer, check expected values."""
    kwargs = cfg_kwargs or DEFAULT_CFG
    r = compile_and_run(source, device=device, max_steps=max_steps,
                        max_seconds=max_seconds, **kwargs)
    for k, v in expected.items():
        if r.get(k) != v:
            mismatches = {k2: (r.get(k2), v2) for k2, v2 in expected.items() if r.get(k2) != v2}
            return False, f"mismatches: {mismatches}"
    return True, f"{r['__num_instructions']:3d} instr, {r['__steps']:5d} steps"


# Level 1: ISA Unit Tests (each opcode individually)

def level1_isa_unit_tests(results, cfg_kw=None):
    print("\n--- Level 1: ISA Unit Tests ---")
    C = cfg_kw or SMALL_CFG
    cfg = Config(**C)
    s, m = cfg.s, cfg.m

    tests = [
        # (name, memory, commands, expected_mem_dict)
        # INC
        ("INC_positive",    [5]+[0]*7,  [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],  {0: 6}),
        ("INC_zero",        [0]+[0]*7,  [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],  {0: 1}),
        ("INC_negative",    [-1]+[0]*7, [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],  {0: 0}),
        ("INC_boundary",    [127]+[0]*7,[(OP_INC, s+0, 0), (OP_HALT, 0, 0)],  {0: -128}),

        # DEC
        ("DEC_positive",    [5]+[0]*7,  [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],  {0: 4}),
        ("DEC_zero",        [0]+[0]*7,  [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],  {0: -1}),
        ("DEC_negative",    [-1]+[0]*7, [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],  {0: -2}),
        ("DEC_boundary",    [-128]+[0]*7,[(OP_DEC, s+0, 0), (OP_HALT, 0, 0)], {0: 127}),

        # MOV
        ("MOV_basic",       [0, 42]+[0]*6, [(OP_MOV, s+0, s+1), (OP_HALT, 0, 0)], {0: 42}),
        ("MOV_negative",    [0, -50]+[0]*6,[(OP_MOV, s+0, s+1), (OP_HALT, 0, 0)], {0: -50}),
        ("MOV_self",        [77]+[0]*7,    [(OP_MOV, s+0, s+0), (OP_HALT, 0, 0)], {0: 77}),

        # ADD
        ("ADD_basic",       [10, 20]+[0]*6, [(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)], {0: 30}),
        ("ADD_negative",    [-10, 20]+[0]*6,[(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)], {0: 10}),
        ("ADD_both_neg",    [-10, -20]+[0]*6,[(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)], {0: -30}),
        ("ADD_zero",        [42, 0]+[0]*6,  [(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)], {0: 42}),
        ("ADD_overflow",    [100, 50]+[0]*6,[(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)], {0: -106}),

        # SUB
        ("SUB_basic",       [10, 3]+[0]*6,  [(OP_SUB, s+0, s+1), (OP_HALT, 0, 0)], {0: 7}),
        ("SUB_negative",    [-10, 3]+[0]*6, [(OP_SUB, s+0, s+1), (OP_HALT, 0, 0)], {0: -13}),
        ("SUB_self",        [42, 42]+[0]*6, [(OP_SUB, s+0, s+1), (OP_HALT, 0, 0)], {0: 0}),

        # AND
        ("AND_basic",       [0b01010101, 0b00110011]+[0]*6, [(OP_AND, s+0, s+1), (OP_HALT, 0, 0)], {0: 0b00010001}),
        ("AND_zero",        [0xFF, 0]+[0]*6,[(OP_AND, s+0, s+1), (OP_HALT, 0, 0)], {0: 0}),
        ("AND_identity",    [42, 127]+[0]*6,[(OP_AND, s+0, s+1), (OP_HALT, 0, 0)], {0: 42}),

        # OR
        ("OR_basic",        [0b01010000, 0b00000101]+[0]*6, [(OP_OR, s+0, s+1), (OP_HALT, 0, 0)], {0: 0b01010101}),
        ("OR_zero",         [42, 0]+[0]*6,  [(OP_OR, s+0, s+1), (OP_HALT, 0, 0)], {0: 42}),

        # XOR
        ("XOR_basic",       [0b01010101, 0b00110011]+[0]*6, [(OP_XOR, s+0, s+1), (OP_HALT, 0, 0)], {0: 0b01100110}),
        ("XOR_self",        [42, 42]+[0]*6, [(OP_XOR, s+0, s+1), (OP_HALT, 0, 0)], {0: 0}),

        # JMP
        ("JMP_forward",     [0]+[0]*7,
         [(OP_JMP, 0, s+m+2), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 0}),

        # JZ (jump if mem[b]==0)
        ("JZ_taken",        [0, 0]+[0]*6,
         [(OP_JZ, s+1, s+m+2), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 0}),
        ("JZ_not_taken",    [0, 5]+[0]*6,
         [(OP_JZ, s+1, s+m+2), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 1}),

        # JNZ (jump if mem[b]!=0)
        ("JNZ_taken",       [0, 5]+[0]*6,
         [(OP_JNZ, s+1, s+m+2), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 0}),
        ("JNZ_not_taken",   [0, 0]+[0]*6,
         [(OP_JNZ, s+1, s+m+2), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 1}),

        # SHL (shift left by 1)
        ("SHL_one",         [1]+[0]*7,  [(OP_SHL, s+0, 0), (OP_HALT, 0, 0)],  {0: 2}),
        ("SHL_three",       [3]+[0]*7,  [(OP_SHL, s+0, 0), (OP_HALT, 0, 0)],  {0: 6}),
        ("SHL_overflow",    [64]+[0]*7, [(OP_SHL, s+0, 0), (OP_HALT, 0, 0)],  {0: -128}),
        ("SHL_neg",         [-1]+[0]*7, [(OP_SHL, s+0, 0), (OP_HALT, 0, 0)],  {0: -2}),
        ("SHL_zero",        [0]+[0]*7,  [(OP_SHL, s+0, 0), (OP_HALT, 0, 0)],  {0: 0}),

        # SHR (arithmetic shift right by 1)
        ("SHR_four",        [4]+[0]*7,  [(OP_SHR, s+0, 0), (OP_HALT, 0, 0)],  {0: 2}),
        ("SHR_seven",       [7]+[0]*7,  [(OP_SHR, s+0, 0), (OP_HALT, 0, 0)],  {0: 3}),
        ("SHR_one",         [1]+[0]*7,  [(OP_SHR, s+0, 0), (OP_HALT, 0, 0)],  {0: 0}),
        ("SHR_neg",         [-2]+[0]*7, [(OP_SHR, s+0, 0), (OP_HALT, 0, 0)],  {0: -1}),
        ("SHR_min",         [-128]+[0]*7,[(OP_SHR, s+0, 0), (OP_HALT, 0, 0)], {0: -64}),
        ("SHR_zero",        [0]+[0]*7,  [(OP_SHR, s+0, 0), (OP_HALT, 0, 0)],  {0: 0}),

        # LOAD: c holds the pointer source; b is the write target.
        ("LOAD_basic",      [2, 0, 42]+[0]*5,
         [(OP_LOAD, s+1, s+0), (OP_HALT, 0, 0)],  {1: 42}),
        ("LOAD_negative",   [3, 0, 0, -7]+[0]*4,
         [(OP_LOAD, s+1, s+0), (OP_HALT, 0, 0)],  {1: -7}),
        ("LOAD_ptr_zero",   [5, 0, 0, 0, 0, 99]+[0]*2,
         [(OP_LOAD, s+1, s+0), (OP_HALT, 0, 0)],  {1: 99}),
        ("LOAD_then_ADD",   [2, 10, 33]+[0]*5,
         [(OP_LOAD, s+3, s+0), (OP_ADD, s+3, s+1), (OP_HALT, 0, 0)],
         {3: 43}),

        # CMP (BLTZ): if mem[b] < 0, PC = c; else PC = PC+1.  mem[b] unchanged.
        ("CMP_negative",    [-5, 0]+[0]*6,
         [(OP_CMP, s+0, s+m+2), (OP_INC, s+1, 0), (OP_HALT, 0, 0)],
         {0: -5, 1: 0}),
        ("CMP_positive",    [5, 0]+[0]*6,
         [(OP_CMP, s+0, s+m+2), (OP_INC, s+1, 0), (OP_HALT, 0, 0)],
         {0: 5, 1: 1}),
        ("CMP_zero",        [0, 0]+[0]*6,
         [(OP_CMP, s+0, s+m+2), (OP_INC, s+1, 0), (OP_HALT, 0, 0)],
         {0: 0, 1: 1}),
        ("CMP_no_modify",   [-128, 0]+[0]*6,
         [(OP_CMP, s+0, s+m+2), (OP_INC, s+1, 0), (OP_HALT, 0, 0)],
         {0: -128, 1: 0}),

        # FIND: mem[dest] = index i where mem[i] == mem[value_addr]
        # FIND requires a unique matching word; duplicate matches average their tags.
        ("FIND_self",       [42]+[0]*7,
         [(OP_FIND, s+1, s+0), (OP_HALT, 0, 0)],  {1: 0}),
        ("FIND_unique",     [0, 10, 0, 0]+[0]*4,
         [(OP_FIND, s+3, s+1), (OP_HALT, 0, 0)],  {3: 1}),
        ("FIND_negative",   [0, 0, 0, 0, -7]+[0]*3,
         [(OP_FIND, s+1, s+4), (OP_HALT, 0, 0)],  {1: 4}),
        ("FIND_last",       [0, 0, 0, 0, 0, 0, 0, 99],
         [(OP_FIND, s+0, s+7), (OP_HALT, 0, 0)],  {0: 7}),

        # CMOV: if mem[b] < 0, mem[b] = mem[c]; else no-op
        ("CMOV_taken",      [-5, 42]+[0]*6,
         [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],  {0: 42}),
        ("CMOV_not_taken",  [5, 42]+[0]*6,
         [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],  {0: 5}),
        ("CMOV_zero",       [0, 99]+[0]*6,
         [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],  {0: 0}),
        ("CMOV_min",        [-128, 0]+[0]*6,
         [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],  {0: 0}),
        ("CMOV_src_intact", [-1, 42]+[0]*6,
         [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],  {0: 42, 1: 42}),

        # MULACC: mem[b] = (mem[b] << 1) + (mem[c] if MSB(mem[b]) else 0).
        ("MULACC_no_msb",   [1, 5]+[0]*6,
         [(OP_MULACC, s+0, s+1), (OP_HALT, 0, 0)],  {0: 2}),
        ("MULACC_msb_set",  [-1, 3]+[0]*6,
         [(OP_MULACC, s+0, s+1), (OP_HALT, 0, 0)],  {0: 1}),
        ("MULACC_min",      [-128, 1]+[0]*6,
         [(OP_MULACC, s+0, s+1), (OP_HALT, 0, 0)],  {0: 1}),
        ("MULACC_zero",     [0, 99]+[0]*6,
         [(OP_MULACC, s+0, s+1), (OP_HALT, 0, 0)],  {0: 0}),
        ("MULACC_shl_only", [64, 7]+[0]*6,
         [(OP_MULACC, s+0, s+1), (OP_HALT, 0, 0)],  {0: -128}),
        ("MULACC_neg_add",  [-64, 10]+[0]*6,
         [(OP_MULACC, s+0, s+1), (OP_HALT, 0, 0)],  {0: -118}),
        ("MULACC_mul_3x5",  [3, 5]+[0]*6,
         [(OP_MULACC, s+0, s+1)] * 8 + [(OP_HALT, 0, 0)],  {0: 15}),
        ("MULACC_mul_7x7",  [7, 7]+[0]*6,
         [(OP_MULACC, s+0, s+1)] * 8 + [(OP_HALT, 0, 0)],  {0: 49}),
        ("MULACC_mul_12x10",[12, 10]+[0]*6,
         [(OP_MULACC, s+0, s+1)] * 8 + [(OP_HALT, 0, 0)],  {0: 120}),
        ("MULACC_mul_2x50", [2, 50]+[0]*6,
         [(OP_MULACC, s+0, s+1)] * 8 + [(OP_HALT, 0, 0)],  {0: 100}),
    ]

    for name, memory, commands, expected in tests:
        try:
            ok, detail = run_isa_test(name, C, memory, commands, expected)
            if ok:
                results.ok(f"L1/{name}")
            else:
                results.fail(f"L1/{name}", detail)
        except Exception as e:
            results.err(f"L1/{name}", e)

    # N=16 tests (16-bit precision)
    print("  --- N=16 precision ---")
    C16 = dict(s=32, m=8, n=64, N=16)
    cfg16 = Config(**C16)
    s16, m16 = cfg16.s, cfg16.m

    tests_16 = [
        ("N16_ADD_large",   [1000, 2000]+[0]*6,
         [(OP_ADD, s16+0, s16+1), (OP_HALT, 0, 0)], {0: 3000}),
        ("N16_ADD_neg",     [-1000, -500]+[0]*6,
         [(OP_ADD, s16+0, s16+1), (OP_HALT, 0, 0)], {0: -1500}),
        ("N16_SUB_large",   [5000, 3000]+[0]*6,
         [(OP_SUB, s16+0, s16+1), (OP_HALT, 0, 0)], {0: 2000}),
        ("N16_SHL_large",   [1024]+[0]*7,
         [(OP_SHL, s16+0, 0), (OP_HALT, 0, 0)], {0: 2048}),
        ("N16_SHR_large",   [4096]+[0]*7,
         [(OP_SHR, s16+0, 0), (OP_HALT, 0, 0)], {0: 2048}),
        ("N16_overflow",    [32000, 1000]+[0]*6,
         [(OP_ADD, s16+0, s16+1), (OP_HALT, 0, 0)], {0: -32536}),
    ]

    for name, memory, commands, expected in tests_16:
        try:
            ok, detail = run_isa_test(name, C16, memory, commands, expected)
            if ok:
                results.ok(f"L1/{name}")
            else:
                results.fail(f"L1/{name}", detail)
        except Exception as e:
            results.err(f"L1/{name}", e)


# Level 2: Multi-Instruction Sequences

def level2_multi_instruction_tests(results, cfg_kw=None):
    print("\n--- Level 2: Multi-Instruction Sequences ---")
    C = cfg_kw or SMALL_CFG
    cfg = Config(**C)
    s, m = cfg.s, cfg.m

    tests = [
        # MOV then ADD
        ("MOV_then_ADD", [0, 10, 20]+[0]*5,
         [(OP_MOV, s+0, s+1), (OP_ADD, s+0, s+2), (OP_HALT, 0, 0)],
         {0: 30}),

        # INC loop (3 increments)
        ("INC_x3", [0]+[0]*7,
         [(OP_INC, s+0, 0), (OP_INC, s+0, 0), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 3}),

        ("INC_then_JZ_skip", [0]+[0]*7,
         [(OP_INC, s+0, 0), (OP_JZ, s+0, s+m+3), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],
         {0: 2}),

        # ADD then SUB (identity)
        ("ADD_then_SUB_identity", [10, 5]+[0]*6,
         [(OP_ADD, s+0, s+1), (OP_SUB, s+0, s+1), (OP_HALT, 0, 0)],
         {0: 10}),

        # SUBLEQ fallback: mem[b] -= mem[a], branch if <=0
        ("SUBLEQ_basic", [3, 10]+[0]*6,
         [(s+0, s+1, s+m+1), (OP_HALT, 0, 0)],  # SUBLEQ: mem[1] -= mem[0] = 7, >0 so no branch
         {0: 3, 1: 7}),

        ("SUBLEQ_branch", [10, 3]+[0]*6,
         [(s+0, s+1, 0), (OP_INC, s+0, 0), (OP_HALT, 0, 0)],  # SUBLEQ: 3-10=-7, <=0 so branch to 0 (halt)
         {0: 10, 1: -7}),
    ]

    for name, memory, commands, expected in tests:
        try:
            ok, detail = run_isa_test(name, C, memory, commands, expected, max_steps=30)
            if ok:
                results.ok(f"L2/{name}")
            else:
                results.fail(f"L2/{name}", detail)
        except Exception as e:
            results.err(f"L2/{name}", e)


# Level 3: Compiled C Programs

# (name, source, expected, heavy)
# heavy=True means the test needs many steps and should only run on CUDA
LEVEL3_PROGRAMS = [
    ("fibonacci", """\
int main() {
    int a = 0; int b = 1; int n = 7; int i = 0;
    while (i < n) { int t = a + b; a = b; b = t; i += 1; }
    return a;
}""", {'__retval': 13}, False),

    ("gcd", """\
int main() {
    int a = 48; int b = 18;
    while (b != 0) { int t = a; while (t >= b) { t = t - b; } a = b; b = t; }
    return a;
}""", {'__retval': 6}, False),

    # Keep the prime-counting loop short; larger ranges require many instructions.
    ("prime_counter_small", """\
int main() {
    int count = 0; int p = 2;
    while (p <= 5) {
        int is_p = 1; int d = 2;
        while (d < p) {
            int rem = p; while (rem >= d) { rem = rem - d; }
            if (rem == 0) { is_p = 0; }
            d += 1;
        }
        if (is_p != 0) { count += 1; }
        p += 1;
    }
    return count;
}""", {'__retval': 3}, True),

    ("bubble_sort_min", """\
int main() {
    int a[6]; a[0]=34; a[1]=12; a[2]=45; a[3]=7; a[4]=23; a[5]=18;
    int n = 6; int i = 0;
    while (i < n) {
        int j = 0;
        while (j < n - 1 - i) {
            int aj = a[j]; int aj1 = a[j+1];
            if (aj > aj1) { a[j] = aj1; a[j+1] = aj; }
            j += 1;
        }
        i += 1;
    }
    return a[0];
}""", {'__retval': 7}, True),

    ("collatz_steps", """\
int main() {
    int n = 6; int steps = 0;
    while (n != 1) {
        int odd = n & 1;
        if (odd != 0) { n = n + n + n + 1; }
        else { int half = 0; int test = 0;
               while (test + test < n) { half += 1; test = half; }
               n = half; }
        steps += 1;
    }
    return steps;
}""", {'__retval': 8}, True),

    ("simple_add", "int main() { return 3 + 5; }", {'__retval': 8}, False),

    ("shl_const", """\
int main() { int x = 3; x = x << 1; return x; }
""", {'__retval': 6}, False),

    ("shr_const", """\
int main() { int x = 12; x = x >> 2; return x; }
""", {'__retval': 3}, False),

    ("shl_multiply", """\
int main() { int x = 5; x = x << 3; return x; }
""", {'__retval': 40}, False),

    ("shr_sign_extend", """\
int main() { int x = -4; x = x >> 1; return x; }
""", {'__retval': -2}, False),

    ("nested_if", """\
int main() {
    int x = 10; int r = 0;
    if (x > 5) { if (x > 8) { r = 2; } else { r = 1; } }
    return r;
}""", {'__retval': 2}, False),

    ("while_sum", """\
int main() {
    int s = 0; int i = 1;
    while (i <= 5) { s += i; i += 1; }
    return s;
}""", {'__retval': 15}, False),

    ("array_sum", """\
int main() {
    int a[4]; a[0]=1; a[1]=2; a[2]=3; a[3]=4;
    int s = a[0] + a[1] + a[2] + a[3];
    return s;
}""", {'__retval': 10}, False),

    ("function_call", """\
int double_it(int x) { return x + x; }
int main() { return double_it(7); }
""", {'__retval': 14}, False),

    ("bitwise_xor_swap", """\
int main() {
    int a = 15; int b = 27;
    a = a ^ b; b = a ^ b; a = a ^ b;
    return a;
}""", {'__retval': 27}, False),

    ("abs_positive", """\
int main() { return abs(7); }
""", {'__retval': 7}, False),

    ("abs_negative", """\
int main() { return abs(-7); }
""", {'__retval': 7}, False),

    ("abs_zero", """\
int main() { return abs(0); }
""", {'__retval': 0}, False),

    ("min_basic", """\
int main() { return min(10, 3); }
""", {'__retval': 3}, False),

    ("min_reverse", """\
int main() { return min(3, 10); }
""", {'__retval': 3}, False),

    ("min_equal", """\
int main() { return min(5, 5); }
""", {'__retval': 5}, False),

    ("max_basic", """\
int main() { return max(3, 10); }
""", {'__retval': 10}, False),

    ("max_reverse", """\
int main() { return max(10, 3); }
""", {'__retval': 10}, False),

    ("array_load_indirect", """\
int main() {
    int a[4]; a[0]=10; a[1]=20; a[2]=30; a[3]=40;
    int i = 2;
    return a[i];
}""", {'__retval': 30}, False),

    ("mul_basic", """\
int main() { return mul(3, 5); }
""", {'__retval': 15}, False),

    ("mul_zero", """\
int main() { return mul(7, 0); }
""", {'__retval': 0}, False),

    ("mul_one", """\
int main() { return mul(42, 1); }
""", {'__retval': 42}, False),

    ("mul_negative", """\
int main() { return mul(-3, 4); }
""", {'__retval': -12}, False),

    ("mul_symmetric", """\
int main() {
    int a = mul(5, 7);
    int b = mul(7, 5);
    if (a != b) { return -1; }
    return a;
}""", {'__retval': 35}, False),

    ("mul_square", """\
int main() { return mul(11, 11); }
""", {'__retval': 121}, False),

    ("sudoku_4x4", """\
int main() {
    int grid[16];
    int pos; int val; int ok; int found;
    int rs; int cs; int bs; int i; int g; int done; int idx;
    grid[0] = -1; grid[1] = 0; grid[2] = 0; grid[3] = -4;
    grid[4] = 0; grid[5] = -4; grid[6] = -1; grid[7] = 0;
    grid[8] = 0; grid[9] = -1; grid[10] = 0; grid[11] = -3;
    grid[12] = -4; grid[13] = 0; grid[14] = 0; grid[15] = -1;
    pos = 0; found = 0;
    while (pos < 16 && found == 0) {
        if (grid[pos] == 0) { found = 1; } else { pos = pos + 1; }
    }
    done = 0;
    while (done == 0) {
        if (pos > 15) { done = 1; }
        else if (pos < 0) { done = 2; }
        else {
            val = grid[pos] + 1; found = 0;
            while (val < 5 && found == 0) {
                ok = 1;
                rs = (pos >> 2) << 2; i = 0;
                while (i < 4) {
                    idx = rs + i;
                    if (idx != pos) { g = grid[idx]; if (g < 0) { g = -g; } if (g == val) { ok = 0; } }
                    i = i + 1;
                }
                if (ok != 0) {
                    cs = pos & 3; i = 0;
                    while (i < 4) {
                        idx = cs + (i << 2);
                        if (idx != pos) { g = grid[idx]; if (g < 0) { g = -g; } if (g == val) { ok = 0; } }
                        i = i + 1;
                    }
                }
                if (ok != 0) {
                    bs = (((pos >> 2) & 2) << 2) + ((pos & 3) & 2); i = 0;
                    while (i < 4) {
                        idx = bs + (i & 1) + ((i >> 1) << 2);
                        if (idx != pos) { g = grid[idx]; if (g < 0) { g = -g; } if (g == val) { ok = 0; } }
                        i = i + 1;
                    }
                }
                if (ok != 0) { found = 1; } else { val = val + 1; }
            }
            if (found != 0) {
                grid[pos] = val; pos = pos + 1; found = 0;
                while (pos < 16 && found == 0) {
                    if (grid[pos] < 0) { pos = pos + 1; } else { found = 1; }
                }
            } else {
                grid[pos] = 0; pos = pos - 1; found = 0;
                while (pos >= 0 && found == 0) {
                    if (grid[pos] < 0) { pos = pos - 1; } else { found = 1; }
                }
            }
        }
    }
    return done;
}""", {'__retval': 1}, True),
]


def level3_compiled_programs(results, device='cpu'):
    print("\n--- Level 3: Compiled C Programs ---")
    for name, source, expected, heavy in LEVEL3_PROGRAMS:
        if heavy and device == 'cpu':
            print(f"  SKIP L3/{name:40s}  (heavy, needs cuda)")
            continue
        try:
            timeout = 300.0 if heavy else 90.0
            ok, detail = run_c_test(name, source, expected, device=device,
                                    max_seconds=timeout)
            if ok:
                results.ok(f"L3/{name}", detail)
            else:
                results.fail(f"L3/{name}", detail)
        except Exception as e:
            results.err(f"L3/{name}", e)

    # N=16 compiler tests (16-bit values beyond 8-bit range)
    print("  --- N=16 compiler ---")
    N16_CFG = dict(s=32, m=64, n=1024, N=16)
    n16_programs = [
        ("N16_add_large", """\
int main() { int a = 1000; int b = 2000; return a + b; }
""", {'__retval': 3000}, False),

        ("N16_fibonacci_15", """\
int main() {
    int a = 0; int b = 1; int n = 15; int i = 0;
    while (i < n) { int t = a + b; a = b; b = t; i += 1; }
    return a;
}""", {'__retval': 610}, True),

        ("N16_shift_large", """\
int main() { int x = 512; x = x << 3; return x; }
""", {'__retval': 4096}, False),
    ]

    for name, source, expected, heavy in n16_programs:
        if heavy and device == 'cpu':
            print(f"  SKIP L3/{name:40s}  (heavy, needs cuda)")
            continue
        try:
            timeout = 300.0 if heavy else 90.0
            ok, detail = run_c_test(name, source, expected, cfg_kwargs=N16_CFG,
                                    device=device, max_seconds=timeout)
            if ok:
                results.ok(f"L3/{name}", detail)
            else:
                results.fail(f"L3/{name}", detail)
        except Exception as e:
            results.err(f"L3/{name}", e)


# Level 4: Scale Tests (same program, different configs)

SCALE_TEST_SOURCE = """\
int main() {
    int a = 0; int b = 1; int n = 7; int i = 0;
    while (i < n) { int t = a + b; a = b; b = t; i += 1; }
    return a;
}"""


def level4_scale_tests(results, device='cpu'):
    print("\n--- Level 4: Scale Tests ---")

    # (name, config, heavy)
    configs = [
        ("scale_1024", dict(s=32, m=64, n=1024, N=8), False),
        ("scale_2048", dict(s=32, m=64, n=2048, N=8), True),
    ]

    retvals = {}
    for name, cfg_kw, heavy in configs:
        if heavy and device == 'cpu':
            print(f"  SKIP L4/{name:40s}  (heavy, needs cuda)")
            continue
        try:
            timeout = 300.0 if heavy else 90.0
            r = compile_and_run(SCALE_TEST_SOURCE, device=device,
                                max_steps=50000, max_seconds=timeout, **cfg_kw)
            retvals[name] = r.get('__retval')
            if r.get('__retval') == 13:
                cfg = Config(**cfg_kw)
                results.ok(f"L4/{name}",
                           f"retval=13, d_model={cfg.d_model}, {r['__steps']} steps")
            else:
                results.fail(f"L4/{name}", f"retval={r.get('__retval')}, expected 13")
        except Exception as e:
            results.err(f"L4/{name}", e)

    # Cross-scale consistency (only if both ran)
    if len(retvals) == 2:
        if retvals.get("scale_1024") == retvals.get("scale_2048"):
            results.ok("L4/cross_scale_consistency", "both configs return same value")
        else:
            results.fail("L4/cross_scale_consistency",
                         f"1024={retvals.get('scale_1024')}, 2048={retvals.get('scale_2048')}")
    else:
        print("  SKIP L4/cross_scale_consistency          (needs both configs)")


# Main

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neural Computer Regression Tests")
    parser.add_argument('device', nargs='?', default='cpu', help='cpu or cuda')
    parser.add_argument('--level', type=int, choices=range(5), default=0,
                        help='Run only this level (1-4; 0=all)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Neural Computer Architecture - Regression Test Suite")
    print(f"Device: {args.device}")
    print("=" * 60)

    results = TestResult()
    level = args.level

    if level == 0 or level == 1:
        level1_isa_unit_tests(results)

    if level == 0 or level == 2:
        level2_multi_instruction_tests(results)

    if level == 0 or level == 3:
        level3_compiled_programs(results, device=args.device)

    if level == 0 or level == 4:
        level4_scale_tests(results, device=args.device)

    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
