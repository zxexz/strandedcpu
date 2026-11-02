#!/usr/bin/env python3
"""Generate raw N=8 reference fixtures and compare them with isa_core.v.

Compares memory, PC, carry, fault, and retired instructions. Compiler check
metadata is omitted.

Run from the repository root:
  python hardware/icestick/test_core.py --generate-only /tmp/core-fixtures.hex
  python hardware/icestick/test_core.py --iverilog /path/to/iverilog --vvp /path/to/vvp
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strandedcpu.compiler import compile_c
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.python import python_to_c


SUPPORTED = set(range(16)) | set(range(17, 22)) | set(range(24, 31)) | set(range(32, 65))


@dataclass
class Case:
    name: str
    memory: list[int]
    commands: list[tuple[int, int, int]]
    limit: int = 1000
    fault: int | None = None
    final: list[int] | None = None
    steps: int | None = None
    pc: int | None = None
    carry: int | None = None

    def reference(self) -> None:
        if self.fault is not None:
            return
        vm = ISAInterpreter(32, 32, 8, self.memory, self.commands)
        while vm.executed < self.limit:
            if vm.step() or vm.pc == 0:
                self.fault = 0
                break
        else:
            self.fault = 3
        self.final = vm.memory
        self.steps, self.pc, self.carry = vm.executed, vm.pc, vm.carry

    def encode(self) -> bytes:
        self.reference()
        assert 1 <= len(self.commands) <= 128
        assert len(self.memory) == len(self.final) == 32
        image = bytearray(512)
        image[:8] = bytes((len(self.commands), self.limit & 255, self.limit >> 8,
                           self.fault, self.steps & 255, self.steps >> 8, self.pc, self.carry))
        image[8:40] = bytes(x & 255 for x in self.memory)
        program = bytes(x for triple in self.commands for x in triple)
        image[40:40+len(program)] = program
        image[424:456] = bytes(x & 255 for x in self.final)
        return bytes(image)


def compiled(name: str, source: str, inputs: dict[str, int] | None = None,
             language: str = "c") -> Case:
    if language == "python":
        source = python_to_c(source)[0]
    cfg, memory, commands, meta = compile_c(source, s=32, m=32, n=256, N=8, checked=True)
    assert cfg.N == 8 and len(commands) <= 128
    assert all(a in SUPPORTED for a, _, _ in commands)
    for key, value in (inputs or {}).items():
        memory[meta["variables"][key]] = value
    return Case(name, memory, commands)


def cases() -> list[Case]:
    result = [
        compiled("C increment", "int x; int y; int main(){y=x+1;return y;}", {"x": 7}),
        compiled("C counter", "int result;int main(){result=0;while(result<10){result+=1;}return result;}"),
        compiled("C checksum", "int data[4]={1,2,3,4};int result;int i;int main(){result=0;for(i=0;i<4;i+=1)result+=data[i];return result;}"),
        compiled("C multiply", "int x;int main(){return x*3+1;}", {"x": 7}),
        compiled("C gcd", "int a=21;int b=15;int main(){while(b!=0){int t=b;b=a%b;a=t;}return a;}"),
        compiled("Python increment", "def solution(x):\n    return x+1\n", {"x": 7}, "python"),
        compiled("Python multiply", "def solution(x):\n    return x*3+1\n", {"x": 7}, "python"),
    ]
    rng = random.Random(20261102)
    bounds = (-128, -127, -9, -8, -1, 0, 1, 7, 8, 127)
    scalar_ops = (1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20,
                  21, 24, 25, 26, 27, 28, 29, 30)
    for op in scalar_ops:
        for k, left in enumerate(bounds):
            right = bounds[(k * 3 + op) % len(bounds)]
            memory = [rng.randrange(-128, 128) for _ in range(32)]
            memory[0], memory[1], memory[2] = left, bounds[(k+1) % len(bounds)], right
            result.append(Case(f"opcode {op} boundary {k}", memory, [(op, 32, 34), (0, 0, 0)]))
    for op in (3, 4, 5, 10, 34):
        for value in bounds:
            memory = [0] * 32
            memory[0], memory[2] = value, 1
            result.append(Case(f"branch {op} value {value}", memory,
                               [(op, 32, 66), (6, 33, 0), (0, 0, 0)]))
    # Carry survives ordinary instructions and feeds ADC. Pair operations
    # read all source values before either word is written.
    memory = [0] * 32
    memory[0], memory[1], memory[2] = -1, 1, 0
    result.append(Case("carry persistence", memory,
                       [(2, 32, 33), (6, 36, 0), (25, 34, 34), (0, 0, 0)]))
    for op in (17, 26, 27):
        memory = [rng.randrange(-128, 128) for _ in range(32)]
        result.append(Case(f"overlapping operands {op}", memory, [(op, 32, 32), (0, 0, 0)]))
    # Indirect jumps hold instruction offsets, not virtual PC addresses.
    memory = [0] * 32
    memory[0] = 2
    result.append(Case("JMPI offset", memory, [(64, 0, 32), (6, 33, 0), (0, 0, 0)]))
    result.append(Case("branch to zero", [0] * 32, [(3, 0, 0)]))
    result.append(Case("full 128-instruction image", [0] * 32,
                       [(6, 32, 0)] * 127 + [(0, 0, 0)]))
    for k in range(96):
        memory = [rng.randrange(-128, 128) for _ in range(32)]
        commands = []
        for i in range(24):
            op = rng.choice(scalar_ops + (3, 4, 5, 10, 32, 33, 34, 63))
            b, c = 32+rng.randrange(28), 32+rng.randrange(28)
            if op in (3, 4, 5, 10) or 32 <= op <= 63:
                c = 64+rng.randrange(i+1, 25)
            commands.append((op, b, c))
        commands.append((0, 0, 0))
        result.append(Case(f"random forward program {k}", memory, commands))
    result.append(Case("step limit", [0] * 32, [(3, 0, 64)], limit=7))
    result.append(Case("step counter exceeds 8 bits", [0] * 32, [(3, 0, 64)], limit=257))
    # The board faults on addresses outside its 32-word memory profile.
    for name, commands, fault, steps, pc, limit in (
        ("unsupported opcode", [(16, 32, 33)], 1, 0, 64, 10),
        ("unsupported page 2", [(65, 32, 33)], 1, 0, 64, 10),
        ("invalid data address", [(1, 31, 32)], 2, 0, 64, 10),
        ("invalid PC", [(3, 0, 65)], 2, 1, 65, 10),
        ("zero step limit", [(0, 0, 0)], 3, 0, 64, 0),
    ):
        result.append(Case(name, [0]*32, commands, limit=limit, fault=fault,
                           final=[0]*32, steps=steps, pc=pc, carry=0))
    return result


def write_fixtures(path: Path, samples: list[Case]) -> None:
    data = b"".join(case.encode() for case in samples)
    path.write_text("\n".join(f"{value:02x}" for value in data) + "\n")
    path.with_suffix(".names.txt").write_text("\n".join(f"{i}: {case.name}" for i, case in enumerate(samples)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-only", type=Path)
    parser.add_argument("--iverilog", default=shutil.which("iverilog"))
    parser.add_argument("--vvp", default=shutil.which("vvp"))
    args = parser.parse_args()
    samples = cases()
    if args.generate_only:
        write_fixtures(args.generate_only, samples)
        print(f"Generated {len(samples)} reference cases: {args.generate_only}")
        return
    if not args.iverilog or not args.vvp:
        parser.error("Icarus Verilog is required; provide --iverilog and --vvp")
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="strandedcpu-core-") as temp:
        folder = Path(temp)
        fixtures = folder / "fixtures.hex"
        binary = folder / "core_tb.vvp"
        write_fixtures(fixtures, samples)
        subprocess.run([args.iverilog, "-g2012", "-s", "core_tb", "-o", str(binary),
                        str(here / "isa_core.v"), str(here / "core_tb.v")], check=True)
        subprocess.run([args.vvp, str(binary), f"+fixtures={fixtures}", f"+count={len(samples)}"], check=True)


if __name__ == "__main__":
    main()
