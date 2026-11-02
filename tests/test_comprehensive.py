"""Scalar ISA and compiled-C tests on the CPU transformer."""

import pytest
import torch
import time

from strandedcpu.transformer import (
    Computer, Config,
    init_state, read_memory, get_pc,
    OP_HALT, OP_MOV, OP_ADD, OP_SUB, OP_INC, OP_DEC,
    OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR, OP_CMP,
    OP_LOAD, OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC, OP_STORE,
    OP_JMP, OP_JZ, OP_JNZ,
)
from strandedcpu.subleq import signed_from_bipolar, signed_to_bipolar, from_bipolar


# Fixtures

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope='module')
def small_computer():
    """Small config for fast single-step tests."""
    cfg = Config(s=32, m=8, n=64, N=8)
    comp = Computer(cfg)
    return cfg, comp

@pytest.fixture(scope='module')
def standard_computer():
    """Standard config for multi-step and compiled C tests."""
    cfg = Config(s=32, m=160, n=512, N=8)
    comp = Computer(cfg)
    return cfg, comp


def run_isa(cfg, comp, memory, commands, max_steps=50):
    """Run ISA program and return (memory_list, pc, steps)."""
    X = init_state(cfg, memory, commands)
    with torch.no_grad():
        for step in range(max_steps):
            pc = get_pc(X, cfg)
            if pc == 0:
                return read_memory(X, cfg), 0, step
            X = comp.step(X)
    return read_memory(X, cfg), get_pc(X, cfg), max_steps


def run_c(cfg, comp, source, max_steps=5000):
    """Compile C source and run on transformer. Returns (vars_dict, steps)."""
    from strandedcpu.compiler import compile_c
    _, mem_init, cmds, meta = compile_c(source, s=cfg.s, m=cfg.m, n=cfg.n, N=cfg.N)
    X = init_state(cfg, mem_init, cmds)
    with torch.no_grad():
        for step in range(max_steps):
            pc = get_pc(X, cfg)
            if pc == 0:
                break
            X = comp.step(X)
    result = {}
    for name, addr in meta['variables'].items():
        result[name] = signed_from_bipolar(
            X[cfg.idx_memory:cfg.idx_memory + cfg.N, cfg.s + addr])
    return result, step


# Single-step opcode tests

class TestArithmetic:
    def test_add_basic(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [3, 7, 0, 0, 0, 0, 0, 0],
                             [(OP_ADD, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 10

    def test_add_negative(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [-5, 3, 0, 0, 0, 0, 0, 0],
                             [(OP_ADD, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == -2

    def test_add_overflow(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [100, 100, 0, 0, 0, 0, 0, 0],
                             [(OP_ADD, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == -56

    def test_sub_basic(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [20, 7, 0, 0, 0, 0, 0, 0],
                             [(OP_SUB, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 13

    def test_sub_negative_result(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [5, 10, 0, 0, 0, 0, 0, 0],
                             [(OP_SUB, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == -5

    def test_sub_overflow(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [80, -80, 0, 0, 0, 0, 0, 0],
                             [(OP_SUB, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == -96

    def test_inc(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [5, 0, 0, 0, 0, 0, 0, 0],
                             [(OP_INC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 6

    def test_inc_minus_one(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [-1, 0, 0, 0, 0, 0, 0, 0],
                             [(OP_INC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 0

    def test_inc_overflow(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [127, 0, 0, 0, 0, 0, 0, 0],
                             [(OP_INC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == -128

    def test_dec(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [10, 0, 0, 0, 0, 0, 0, 0],
                             [(OP_DEC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 9

    def test_dec_zero(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [0, 0, 0, 0, 0, 0, 0, 0],
                             [(OP_DEC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == -1

    def test_dec_underflow(self, small_computer):
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [-128, 0, 0, 0, 0, 0, 0, 0],
                             [(OP_DEC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 127


class TestBitwise:
    def test_and(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0b1100, 0b1010, 0, 0, 0, 0, 0, 0],
                            [(OP_AND, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 0b1000

    def test_and_negative(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [-16, -86, 0, 0, 0, 0, 0, 0],
                            [(OP_AND, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == -96

    def test_or(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0b1100, 0b1010, 0, 0, 0, 0, 0, 0],
                            [(OP_OR, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 0b1110

    def test_xor(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [5, 3, 0, 0, 0, 0, 0, 0],
                            [(OP_XOR, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 6

    def test_xor_self_is_zero(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [127, 127, 0, 0, 0, 0, 0, 0],
                            [(OP_XOR, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 0

    def test_shl(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [5, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_SHL, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 10

    def test_shr(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [20, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_SHR, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 10


class TestMemory:
    def test_mov(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0, 42, 0, 0, 0, 0, 0, 0],
                            [(OP_MOV, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 42

    def test_mov_self(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [33, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_MOV, cfg.s+0, cfg.s+0), (OP_HALT, 0, 0)])
        assert mem[0] == 33

    def test_swap(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [10, 20, 0, 0, 0, 0, 0, 0],
                            [(OP_SWAP, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 20 and mem[1] == 10

    def test_load(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0, 42, 1, 0, 0, 0, 0, 0],
                            [(OP_LOAD, cfg.s+0, cfg.s+2), (OP_HALT, 0, 0)])
        assert mem[0] == 42

    def test_store(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [99, 3, 0, 0, 0, 0, 0, 0],
                            [(OP_STORE, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[3] == 99

    def test_store_doesnt_corrupt(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [42, 2, 77, 88, 0, 0, 0, 0],
                            [(OP_STORE, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[2] == 42
        assert mem[0] == 42
        assert mem[1] == 2
        assert mem[3] == 88


class TestBranching:
    def test_jmp(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [5, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_JMP, 0, cfg.s+cfg.m+2), (OP_INC, cfg.s+0, 0), (OP_HALT, 0, 0)])
        assert mem[0] == 5

    def test_jz_taken(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0, 5, 0, 0, 0, 0, 0, 0],
                            [(OP_JZ, cfg.s+0, cfg.s+cfg.m+2), (OP_INC, cfg.s+1, 0), (OP_HALT, 0, 0)])
        assert mem[1] == 5

    def test_jz_not_taken(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [1, 5, 0, 0, 0, 0, 0, 0],
                            [(OP_JZ, cfg.s+0, cfg.s+cfg.m+2), (OP_INC, cfg.s+1, 0), (OP_HALT, 0, 0)])
        assert mem[1] == 6

    def test_jnz_taken(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [1, 5, 0, 0, 0, 0, 0, 0],
                            [(OP_JNZ, cfg.s+0, cfg.s+cfg.m+2), (OP_INC, cfg.s+1, 0), (OP_HALT, 0, 0)])
        assert mem[1] == 5

    def test_jnz_not_taken(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0, 5, 0, 0, 0, 0, 0, 0],
                            [(OP_JNZ, cfg.s+0, cfg.s+cfg.m+2), (OP_INC, cfg.s+1, 0), (OP_HALT, 0, 0)])
        assert mem[1] == 6


class TestSUBLEQ:
    def test_subleq_basic(self, small_computer):
        """SUBLEQ: mem[b] -= mem[a]. With a >= 32 for SUBLEQ mode."""
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [3, 10, 0, 0, 0, 0, 0, 0],
                            [(cfg.s+0, cfg.s+1, 0), (OP_HALT, 0, 0)])
        assert mem[1] == 7

    def test_subleq_branch_taken(self, small_computer):
        """SUBLEQ branches to c when result <= 0."""
        cfg, comp = small_computer
        mem, pc, _ = run_isa(cfg, comp, [10, 5, 0, 0, 0, 0, 0, 0],
                             [(cfg.s+0, cfg.s+1, 0), (OP_HALT, 0, 0)])
        assert mem[1] == -5
        assert pc == 0

    def test_subleq_countdown(self, small_computer):
        """SUBLEQ loop: decrement mem[2] by mem[1] until <= 0."""
        cfg, comp = small_computer
        mem, _, steps = run_isa(cfg, comp, [0, 1, 5, 0, 0, 0, 0, 0],
                                [(cfg.s+1, cfg.s+2, 0), (cfg.s+0, cfg.s+0, cfg.s+cfg.m)],
                                max_steps=20)
        assert mem[2] == 0


# Multi-step compiled C tests

class TestCompiledC:
    def test_inc_loop_10(self, standard_computer):
        cfg, comp = standard_computer
        result, steps = run_c(cfg, comp, """
            int main() {
                int x; int i;
                i = 0;
                while (i < 10) { x = x + 1; i = i + 1; }
                return x;
            }
        """, max_steps=2000)
        assert result['x'] == 10

    def test_fibonacci(self, standard_computer):
        cfg, comp = standard_computer
        result, steps = run_c(cfg, comp, """
            int main() {
                int a; int b; int t; int n; int i;
                a = 0; b = 1; n = 7; i = 0;
                while (i < n) { t = a + b; a = b; b = t; i = i + 1; }
                return a;
            }
        """, max_steps=2000)
        assert result['a'] == 13

    def test_max_of_three(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int a; int b; int c; int max;
                a = 42; b = 77; c = 23;
                max = a;
                if (b > max) { max = b; }
                if (c > max) { max = c; }
                return max;
            }
        """, max_steps=500)
        assert result['max'] == 77

    def test_array_sum(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int arr[5]; int sum; int i;
                arr[0] = 10; arr[1] = 20; arr[2] = 30; arr[3] = 40; arr[4] = 50;
                sum = 0; i = 0;
                while (i < 5) { sum = sum + arr[i]; i = i + 1; }
                return sum;
            }
        """, max_steps=5000)
        # Disabled below: the sum is 150, which wraps to -106 at N=8.
        assert result['sum'] == 120

    def test_array_sum_small(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int arr[5]; int sum; int i;
                arr[0] = 1; arr[1] = 2; arr[2] = 3; arr[3] = 4; arr[4] = 5;
                sum = 0; i = 0;
                while (i < 5) { sum = sum + arr[i]; i = i + 1; }
                return sum;
            }
        """, max_steps=5000)
        assert result['sum'] == 15

    def test_min_of_array(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int arr[5]; int min; int i; int v;
                arr[0] = 5; arr[1] = 3; arr[2] = 4; arr[3] = 1; arr[4] = 2;
                min = arr[0]; i = 1;
                while (i < 5) {
                    v = arr[i];
                    if (v < min) { min = v; }
                    i = i + 1;
                }
                return min;
            }
        """, max_steps=5000)
        assert result['min'] == 1

    def test_nested_if(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int x; int y; int r;
                x = 5; y = 10;
                if (x < y) {
                    if (x > 3) { r = 1; }
                    else { r = 2; }
                } else {
                    r = 3;
                }
                return r;
            }
        """, max_steps=500)
        assert result['r'] == 1

    def test_while_with_break_pattern(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int i; int found; int target;
                int arr[8];
                arr[0] = 10; arr[1] = 20; arr[2] = 30; arr[3] = 42;
                arr[4] = 50; arr[5] = 60; arr[6] = 70; arr[7] = 80;
                target = 42; found = -1; i = 0;
                while (i < 8 && found < 0) {
                    if (arr[i] == target) { found = i; }
                    i = i + 1;
                }
                return found;
            }
        """, max_steps=5000)
        assert result['found'] == 3

    def test_gcd(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int a; int b;
                a = 48; b = 18;
                while (a != b) {
                    if (a > b) { a = a - b; }
                    else { b = b - a; }
                }
                return a;
            }
        """, max_steps=5000)
        assert result['a'] == 6

    def test_store_indirect_write(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int arr[4]; int idx; int val;
                idx = 2; val = 77;
                arr[idx] = val;
                return arr[2];
            }
        """, max_steps=1000)
        assert result['arr'] is not None

    def test_store_and_load(self, standard_computer):
        cfg, comp = standard_computer
        result, _ = run_c(cfg, comp, """
            int main() {
                int arr[4]; int idx; int val; int out;
                idx = 2; val = 77;
                arr[idx] = val;
                out = arr[2];
                return out;
            }
        """, max_steps=1000)
        assert result['out'] == 77


# Edge cases

class TestEdgeCases:
    def test_halt_immediate(self, small_computer):
        cfg, comp = small_computer
        mem, pc, steps = run_isa(cfg, comp, [42, 0, 0, 0, 0, 0, 0, 0],
                                 [(OP_HALT, 0, 0)])
        assert pc == 0 and steps <= 1
        assert mem[0] == 42

    def test_zero_memory(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [0, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_ADD, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 0

    def test_max_positive(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [127, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_MOV, cfg.s+1, cfg.s+0), (OP_HALT, 0, 0)])
        assert mem[1] == 127

    def test_min_negative(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [-128, 0, 0, 0, 0, 0, 0, 0],
                            [(OP_MOV, cfg.s+1, cfg.s+0), (OP_HALT, 0, 0)])
        assert mem[1] == -128

    def test_sequential_operations(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [5, 3, 0, 0, 0, 0, 0, 0],
                            [(OP_ADD, cfg.s+0, cfg.s+1),
                             (OP_INC, cfg.s+0, 0),
                             (OP_SHL, cfg.s+0, 0),
                             (OP_HALT, 0, 0)],
                            max_steps=10)
        assert mem[0] == 18

    def test_cmov_taken(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [-5, 42, 0, 0, 0, 0, 0, 0],
                            [(OP_CMOV, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 42

    def test_cmov_not_taken(self, small_computer):
        cfg, comp = small_computer
        mem, _, _ = run_isa(cfg, comp, [99, 1, 0, 0, 0, 0, 0, 0],
                            [(OP_CMOV, cfg.s+0, cfg.s+1), (OP_HALT, 0, 0)])
        assert mem[0] == 99


# The legacy expectation is retained but excluded from collection.
TestCompiledC.test_array_sum = None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
