"""Compare base and broadcast-bias transformers on scalar ISA programs.

Both standard-layer entry points use hard top-two attention.
"""

import torch
import sys
from strandedcpu.standard import (
    StandardComputer, StandardConfig,
    init_state_standard, read_memory_standard, get_pc_standard,
    OP_HALT, OP_MOV, OP_ADD, OP_JMP, OP_JZ, OP_JNZ, OP_INC, OP_DEC,
    OP_SHL, OP_SHR, OP_CMP, OP_LOAD, OP_AND, OP_OR, OP_XOR, OP_SUB,
    OP_FIND, OP_SWAP, OP_CMOV, OP_MULACC,
)
from strandedcpu.transformer import (
    Computer, Config,
    init_state, read_memory, get_pc,
)


def run_base(cfg_base, memory, commands, max_steps=20):
    """Run the base transformer."""
    ext = Computer(cfg_base)
    X = init_state(cfg_base, memory, commands)
    with torch.no_grad():
        for _ in range(max_steps):
            pc = get_pc(X, cfg_base)
            if pc == 0:
                break
            X = ext.step(X)
    return read_memory(X, cfg_base)


def run_std(cfg_std, ext_std, memory, commands, max_steps=20):
    """Run the broadcast-bias transformer through its default entry point."""
    X = init_state_standard(cfg_std, memory, commands)
    with torch.no_grad():
        for _ in range(max_steps):
            pc = get_pc_standard(X, cfg_std)
            if pc == 0:
                break
            X = ext_std.step(X)
    return read_memory_standard(X, cfg_std)


def run_argmax(cfg_std, memory, commands, max_steps=20):
    """Run the broadcast-bias transformer through its argmax alias."""
    ext = StandardComputer(cfg_std, use_argmax=True)
    X = init_state_standard(cfg_std, memory, commands)
    with torch.no_grad():
        for _ in range(max_steps):
            pc = get_pc_standard(X, cfg_std)
            if pc == 0:
                break
            X = ext.step(X)
    return read_memory_standard(X, cfg_std)


def check_opcode(name, cfg_base, cfg_std, ext_std, memory, commands, expected,
                max_steps=20, test_argmax=True):
    """Test one opcode: base transformer, standard, and optionally argmax."""
    s = cfg_base.s
    m_count = len(expected)

    base_result = run_base(cfg_base, memory, commands, max_steps)[:m_count]
    std_result = run_std(cfg_std, ext_std, memory, commands, max_steps)[:m_count]

    base_ok = base_result == expected
    std_ok = std_result == expected

    status = "PASS" if (base_ok and std_ok) else "FAIL"
    print(f"  [{status}] {name}: expected={expected} base={base_result} std={std_result}", end="")

    arg_ok = True
    if test_argmax:
        arg_result = run_argmax(cfg_std, memory, commands, max_steps)[:m_count]
        arg_ok = arg_result == expected
        print(f" argmax={arg_result}", end="")
        if not arg_ok:
            status = "FAIL(argmax)"
    print()

    return base_ok and std_ok and arg_ok


def main():
    cfg_base = Config(s=32, m=8, n=64, N=8)
    cfg_std = StandardConfig(s=32, m=8, n=64, N=8)
    ext_std = StandardComputer(cfg_std)
    s = cfg_base.s

    tests_passed = 0
    tests_total = 0

    def T(name, memory, commands, expected, max_steps=20, test_argmax=True):
        nonlocal tests_passed, tests_total
        tests_total += 1
        if check_opcode(name, cfg_base, cfg_std, ext_std, memory, commands,
                       expected, max_steps, test_argmax):
            tests_passed += 1

    print("=" * 60)
    print("base transformer Standard Test Suite")
    print("=" * 60)

    # Basic SUBLEQ
    print("\n-- SUBLEQ --")
    T("SUBLEQ 5-3=2",
      [5, 3, 0, 0, 0, 0, 0, 0],
      [(s+1, s+0, 0)],  # mem[0] -= mem[1]; halt if <= 0
      [2, 3, 0, 0, 0, 0, 0, 0])

    T("SUBLEQ 5-3-3=-1 halt",
      [5, 3, 0, 0, 0, 0, 0, 0],
      [(s+1, s+0, 0),         # mem[0] -= mem[1]; if <= 0 goto 0 (halt)
       (s+1, s+0, 0)],
      [-1, 3, 0, 0, 0, 0, 0, 0],
      max_steps=5)

    # INC / DEC
    print("\n-- INC / DEC --")
    T("INC 5->6",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],
      [6, 0, 0, 0, 0, 0, 0, 0])

    T("DEC 5->4",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],
      [4, 0, 0, 0, 0, 0, 0, 0])

    T("INC 0->1",
      [0, 0, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],
      [1, 0, 0, 0, 0, 0, 0, 0])

    T("DEC 0->-1",
      [0, 0, 0, 0, 0, 0, 0, 0],
      [(OP_DEC, s+0, 0), (OP_HALT, 0, 0)],
      [-1, 0, 0, 0, 0, 0, 0, 0])

    T("INC -1->0",
      [-1, 0, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0), (OP_HALT, 0, 0)],
      [0, 0, 0, 0, 0, 0, 0, 0])

    # MOV
    print("\n-- MOV --")
    T("MOV 0<-7",
      [0, 7, 0, 0, 0, 0, 0, 0],
      [(OP_MOV, s+0, s+1), (OP_HALT, 0, 0)],
      [7, 7, 0, 0, 0, 0, 0, 0])

    T("MOV 0<-(-3)",
      [0, -3, 0, 0, 0, 0, 0, 0],
      [(OP_MOV, s+0, s+1), (OP_HALT, 0, 0)],
      [-3, -3, 0, 0, 0, 0, 0, 0])

    # ADD / SUB
    print("\n-- ADD / SUB --")
    T("ADD 3+5=8",
      [3, 5, 0, 0, 0, 0, 0, 0],
      [(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)],
      [8, 5, 0, 0, 0, 0, 0, 0])

    T("ADD 3+(-2)=1",
      [3, -2, 0, 0, 0, 0, 0, 0],
      [(OP_ADD, s+0, s+1), (OP_HALT, 0, 0)],
      [1, -2, 0, 0, 0, 0, 0, 0])

    T("SUB 8-3=5",
      [8, 3, 0, 0, 0, 0, 0, 0],
      [(OP_SUB, s+0, s+1), (OP_HALT, 0, 0)],
      [5, 3, 0, 0, 0, 0, 0, 0])

    # JMP / JZ / JNZ
    print("\n-- JMP / JZ / JNZ --")
    T("JMP skip",
      [10, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JMP, 0, s+8+2),
       (OP_INC, s+0, 0),
       (OP_HALT, 0, 0)],
      [10, 0, 0, 0, 0, 0, 0, 0])

    T("JZ taken (val=0)",
      [0, 99, 0, 0, 0, 0, 0, 0],
      [(OP_JZ, s+0, s+8+2),
       (OP_INC, s+1, 0),
       (OP_HALT, 0, 0)],
      [0, 99, 0, 0, 0, 0, 0, 0])

    T("JZ not taken (val=5)",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JZ, s+0, s+8+3),
       (OP_INC, s+1, 0),
       (OP_HALT, 0, 0)],
      [5, 1, 0, 0, 0, 0, 0, 0])

    T("JNZ taken (val=5)",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JNZ, s+0, s+8+2),
       (OP_INC, s+1, 0),
       (OP_HALT, 0, 0)],
      [5, 0, 0, 0, 0, 0, 0, 0])

    T("JNZ not taken (val=0)",
      [0, 0, 0, 0, 0, 0, 0, 0],
      [(OP_JNZ, s+0, s+8+3),
       (OP_INC, s+1, 0),
       (OP_HALT, 0, 0)],
      [0, 1, 0, 0, 0, 0, 0, 0])

    # Bitwise
    print("\n-- AND / OR / XOR --")
    T("AND 0b1100 & 0b1010 = 0b1000",
      [12, 10, 0, 0, 0, 0, 0, 0],
      [(OP_AND, s+0, s+1), (OP_HALT, 0, 0)],
      [8, 10, 0, 0, 0, 0, 0, 0])

    T("OR 0b1100 | 0b1010 = 0b1110",
      [12, 10, 0, 0, 0, 0, 0, 0],
      [(OP_OR, s+0, s+1), (OP_HALT, 0, 0)],
      [14, 10, 0, 0, 0, 0, 0, 0])

    T("XOR 0b1100 ^ 0b1010 = 0b0110",
      [12, 10, 0, 0, 0, 0, 0, 0],
      [(OP_XOR, s+0, s+1), (OP_HALT, 0, 0)],
      [6, 10, 0, 0, 0, 0, 0, 0])

    # SHL / SHR
    print("\n-- SHL / SHR --")
    T("SHL 5<<1=10",
      [0, 5, 0, 0, 0, 0, 0, 0],
      [(OP_SHL, s+1, 0), (OP_HALT, 0, 0)],
      [0, 10, 0, 0, 0, 0, 0, 0])

    T("SHR 10>>1=5",
      [0, 10, 0, 0, 0, 0, 0, 0],
      [(OP_SHR, s+1, 0), (OP_HALT, 0, 0)],
      [0, 5, 0, 0, 0, 0, 0, 0])

    # CMP
    print("\n-- CMP --")
    T("CMP neg branches",
      [-1, 0, 0, 0, 0, 0, 0, 0],
      [(OP_CMP, s+0, s+8+2),
       (OP_INC, s+1, 0),
       (OP_HALT, 0, 0)],
      [-1, 0, 0, 0, 0, 0, 0, 0])

    T("CMP pos falls through",
      [5, 0, 0, 0, 0, 0, 0, 0],
      [(OP_CMP, s+0, s+8+3),
       (OP_INC, s+1, 0),
       (OP_HALT, 0, 0)],
      [5, 1, 0, 0, 0, 0, 0, 0])

    # SWAP
    print("\n-- SWAP --")
    T("SWAP 3,7",
      [3, 7, 0, 0, 0, 0, 0, 0],
      [(OP_SWAP, s+0, s+1), (OP_HALT, 0, 0)],
      [7, 3, 0, 0, 0, 0, 0, 0])

    # CMOV
    print("\n-- CMOV --")
    T("CMOV taken (neg)",
      [-1, 42, 0, 0, 0, 0, 0, 0],
      [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],
      [42, 42, 0, 0, 0, 0, 0, 0])

    T("CMOV not taken (pos)",
      [5, 42, 0, 0, 0, 0, 0, 0],
      [(OP_CMOV, s+0, s+1), (OP_HALT, 0, 0)],
      [5, 42, 0, 0, 0, 0, 0, 0])

    # Multi-step programs
    print("\n-- Multi-step --")
    T("INC loop 3x",
      [0, 3, 0, 0, 0, 0, 0, 0],
      [(OP_INC, s+0, 0),
       (OP_DEC, s+1, 0),
       (OP_JNZ, s+1, s+8+0),
       (OP_HALT, 0, 0)],
      [3, 0, 0, 0, 0, 0, 0, 0],
      max_steps=50)

    T("MOV+ADD: a=b+c",
      [0, 3, 5, 0, 0, 0, 0, 0],
      [(OP_MOV, s+0, s+1),
       (OP_ADD, s+0, s+2),
       (OP_HALT, 0, 0)],
      [8, 3, 5, 0, 0, 0, 0, 0])

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {tests_passed}/{tests_total} passed")
    if tests_passed == tests_total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {tests_total - tests_passed}")
    print(f"{'='*60}")

    return tests_passed == tests_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
