"""Test scalar SWAP and compiled C programs that exchange values."""

import torch
import pytest
from strandedcpu.transformer import (
    Computer, Config,
    init_state, read_memory, get_pc,
    OP_HALT, OP_SWAP, OP_MOV, OP_INC, OP_ADD, OP_SUB,
)


def make_cfg(m=8):
    return Config(s=32, m=m, n=64, N=8)


def run_program(cfg, memory, commands, max_steps=100):
    """Run a program and return final memory."""
    comp = Computer(cfg)
    X = init_state(cfg, memory, commands)
    with torch.no_grad():
        X, steps = comp.run(X, max_steps=max_steps)
    return read_memory(X, cfg), steps


# Basic SWAP tests

class TestSwapBasic:

    def test_swap_positive_values(self):
        cfg = make_cfg()
        mem = [10, 20] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 20, f"mem[0] should be 20, got {result[0]}"
        assert result[1] == 10, f"mem[1] should be 10, got {result[1]}"

    def test_swap_positive_and_negative(self):
        cfg = make_cfg()
        mem = [42, -17] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == -17, f"mem[0] should be -17, got {result[0]}"
        assert result[1] == 42, f"mem[1] should be 42, got {result[1]}"

    def test_swap_both_negative(self):
        cfg = make_cfg()
        mem = [-5, -100] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == -100, f"mem[0] should be -100, got {result[0]}"
        assert result[1] == -5, f"mem[1] should be -5, got {result[1]}"

    def test_swap_with_zero(self):
        cfg = make_cfg()
        mem = [0, 77] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 77, f"mem[0] should be 77, got {result[0]}"
        assert result[1] == 0, f"mem[1] should be 0, got {result[1]}"

    def test_swap_same_value(self):
        cfg = make_cfg()
        mem = [33, 33] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 33
        assert result[1] == 33

    def test_swap_same_cell(self):
        cfg = make_cfg()
        mem = [42] + [0] * (cfg.m - 1)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 0),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 42, f"mem[0] should be 42, got {result[0]}"


# SWAP with other opcodes

class TestSwapIntegration:

    def test_swap_then_add(self):
        cfg = make_cfg()
        mem = [10, 20, 0] + [0] * (cfg.m - 3)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_ADD, cfg.s + 2, cfg.s + 0),
            (OP_ADD, cfg.s + 2, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 20
        assert result[1] == 10
        assert result[2] == 30

    def test_double_swap_identity(self):
        cfg = make_cfg()
        mem = [7, 99] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 7, f"Double SWAP should be identity: got {result[0]}"
        assert result[1] == 99, f"Double SWAP should be identity: got {result[1]}"

    def test_swap_doesnt_corrupt_other_memory(self):
        cfg = make_cfg()
        mem = [11, 22, 33, 44] + [0] * (cfg.m - 4)
        cmds = [
            (OP_SWAP, cfg.s + 1, cfg.s + 2),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 11, f"mem[0] should be unchanged: {result[0]}"
        assert result[1] == 33, f"mem[1] should be 33: {result[1]}"
        assert result[2] == 22, f"mem[2] should be 22: {result[2]}"
        assert result[3] == 44, f"mem[3] should be unchanged: {result[3]}"

    def test_swap_non_adjacent(self):
        cfg = make_cfg()
        mem = [1, 0, 0, 0, 5] + [0] * (cfg.m - 5)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 4),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 5
        assert result[4] == 1

    def test_inc_after_swap(self):
        cfg = make_cfg()
        mem = [10, 20] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_INC, cfg.s + 0, 0),
            (OP_INC, cfg.s + 1, 0),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 21
        assert result[1] == 11

    def test_three_way_rotate(self):
        cfg = make_cfg()
        mem = [1, 2, 3] + [0] * (cfg.m - 3)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_SWAP, cfg.s + 1, cfg.s + 2),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 2
        assert result[1] == 3
        assert result[2] == 1


# Edge cases

class TestSwapEdgeCases:

    def test_swap_max_positive(self):
        cfg = make_cfg()
        mem = [127, 1] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 1
        assert result[1] == 127

    def test_swap_min_negative(self):
        cfg = make_cfg()
        mem = [-128, 0] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 0
        assert result[1] == -128

    def test_swap_minus_one(self):
        cfg = make_cfg()
        mem = [-1, 1] + [0] * (cfg.m - 2)
        cmds = [
            (OP_SWAP, cfg.s + 0, cfg.s + 1),
            (OP_HALT, 0, 0),
        ]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 1
        assert result[1] == -1


# Arithmetic after the two-head L5 write stage

class TestSwapRegression:

    def test_inc_still_works(self):
        cfg = make_cfg()
        mem = [5] + [0] * (cfg.m - 1)
        cmds = [(OP_INC, cfg.s + 0, 0), (OP_HALT, 0, 0)]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 6

    def test_mov_still_works(self):
        cfg = make_cfg()
        mem = [0, 42] + [0] * (cfg.m - 2)
        cmds = [(OP_MOV, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 42

    def test_add_still_works(self):
        cfg = make_cfg()
        mem = [10, 20] + [0] * (cfg.m - 2)
        cmds = [(OP_ADD, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 30

    def test_sub_still_works(self):
        cfg = make_cfg()
        mem = [30, 12] + [0] * (cfg.m - 2)
        cmds = [(OP_SUB, cfg.s + 0, cfg.s + 1), (OP_HALT, 0, 0)]
        result, _ = run_program(cfg, mem, cmds)
        assert result[0] == 18


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
