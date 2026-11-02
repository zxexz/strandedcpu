"""UART client for the iCEstick raw 8-bit ISA profile.

Execution wraps at eight bits. Compiler runtime-check metadata is not sent
to the board, so this profile does not provide checked option scoring.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import struct
import time
from typing import Mapping

from strandedcpu.compiler import TT, compile_c, lex
from strandedcpu.interpreter import ISAInterpreter
from strandedcpu.python import python_to_c


MEMORY_WORDS = 32
PROGRAM_SLOTS = 128
SUPPORTED_OPCODES = frozenset(range(16)) | set(range(17, 22)) | set(range(24, 31)) | frozenset(range(32, 65))
FAULTS = {0: "halt", 1: "unsupported_opcode", 2: "invalid_address", 3: "step_limit"}


@dataclass(frozen=True)
class Program:
    memory: tuple[int, ...]
    commands: tuple[tuple[int, int, int], ...]
    metadata: dict
    result: str = "return"

    def payload(self) -> bytes:
        if len(self.memory) != MEMORY_WORDS:
            raise ValueError("iCEstick requires exactly 32 memory words")
        if not 1 <= len(self.commands) <= PROGRAM_SLOTS:
            raise ValueError("iCEstick supports 1..128 instructions")
        for command in self.commands:
            if len(command) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in command):
                raise ValueError("instruction fields must be unsigned bytes")
            if command[0] not in SUPPORTED_OPCODES:
                raise ValueError(f"opcode {command[0]} is outside the iCEstick profile")
        if any(not isinstance(v, int) or not -128 <= v <= 127 for v in self.memory):
            raise ValueError("memory words must be signed 8-bit integers")
        return bytes(v & 255 for v in self.memory) + bytes([len(self.commands)]) + bytes(
            v for command in self.commands for v in command
        )

    def variables(self, memory: tuple[int, ...]) -> dict:
        values = {}
        for name, addr in self.metadata["variables"].items():
            count = self.metadata["sizes"].get(name, 1)
            values[name] = memory[addr] if count == 1 else list(memory[addr:addr + count])
        values["__retval"] = memory[self.metadata["retval_addr"]]
        return values


def compile_program(source: str, inputs: Mapping[str, int] | None = None, *, language: str = "c") -> Program:
    """Compile an integer program for N=8, s=32, m=32, n=256.

    Literals and host inputs must fit. Execution uses raw wrapping arithmetic;
    metadata checks are not uploaded or enforced. Fixed point is unsupported.
    """
    if language == "python":
        tree = ast.parse(source)
        if any((isinstance(node, ast.Constant) and isinstance(node.value, float))
               or isinstance(node, ast.Div)
               or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float")
               for node in ast.walk(tree)):
            raise ValueError("the iCEstick profile supports integer programs only")
        source = python_to_c(source)[0]
        result = "result"
    elif language == "c":
        if any(token.type == TT.FIX or (token.type == TT.NUMBER and "." in token.value) for token in lex(source)):
            raise ValueError("the iCEstick profile supports integer programs only")
        result = "return"
    else:
        raise ValueError("language must be 'c' or 'python'")
    _, memory, commands, metadata = compile_c(source, s=32, m=32, n=256, N=8, checked=True)
    if metadata["fix_variables"]:
        raise ValueError("the iCEstick profile supports integer programs only")
    for name, value in (inputs or {}).items():
        if name not in metadata["variables"] or metadata["sizes"].get(name, 1) != 1:
            raise ValueError(f"no scalar input global {name!r}")
        if isinstance(value, bool) or not isinstance(value, int) or not -128 <= value <= 127:
            raise ValueError(f"{name} must be a signed 8-bit integer")
        memory[metadata["variables"][name]] = value
    image = Program(tuple(memory), tuple(tuple(c) for c in commands), metadata, result)
    image.payload()
    return image


@dataclass(frozen=True)
class Result:
    fault: int
    steps: int
    pc: int
    carry: int
    memory: tuple[int, ...]

    def as_dict(self, program: Program) -> dict:
        values = program.variables(self.memory)
        return {
            "backend": "icestick", "profile": "raw-int8", "checked": False,
            "status": FAULTS.get(self.fault, f"fault_{self.fault}"),
            "steps": self.steps, "pc": self.pc, "carry": self.carry,
            "value": values["__retval"] if program.result == "return" else values[program.result],
            "variables": values, "memory": list(self.memory),
        }


class SerialTransport:
    """POSIX serial transport at 115200 baud."""

    def __init__(self, port: str, timeout: float = 5.0):
        import termios
        self.timeout = timeout
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(self.fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[4] = attrs[5] = termios.B115200
            attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        os.close(self.fd)

    def write(self, data: bytes):
        pending = memoryview(data)
        deadline = time.monotonic() + self.timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                raise TimeoutError("iCEstick serial write timed out")
            try:
                count = os.write(self.fd, pending)
            except BlockingIOError:
                continue
            if count <= 0:
                raise OSError("iCEstick serial port closed during write")
            pending = pending[count:]

    def read_exact(self, count: int) -> bytes:
        data = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(data) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                raise TimeoutError(f"iCEstick reply timed out ({len(data)}/{count} bytes)")
            try:
                block = os.read(self.fd, count - len(data))
            except BlockingIOError:
                continue
            if not block:
                raise OSError("iCEstick serial port closed during read")
            data.extend(block)
        return bytes(data)


class IceStick:
    def __init__(self, transport):
        self.transport = transport

    @classmethod
    def open(cls, port: str, timeout: float = 5.0):
        return cls(SerialTransport(port, timeout))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.transport.close()

    def ping(self):
        self.transport.write(b"P")
        reply = self.transport.read_exact(4)
        if reply != b"SC8\x01":
            raise RuntimeError(f"unexpected iCEstick firmware identifier: {reply!r}")
        return reply

    def load(self, program: Program):
        self.transport.write(b"L" + program.payload())
        if self.transport.read_exact(1) != b"K":
            raise RuntimeError("iCEstick rejected the program upload")

    def run(self, max_steps: int = 65535) -> Result:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 65535:
            raise ValueError("max_steps must be 1..65535")
        self.transport.write(b"R" + struct.pack("<H", max_steps))
        header = self.transport.read_exact(6)
        if header[0] != ord("D"):
            raise RuntimeError(f"invalid iCEstick result packet: {header!r}")
        data = self.transport.read_exact(MEMORY_WORDS)
        memory = tuple(v - 256 if v >= 128 else v for v in data)
        return Result(header[1], int.from_bytes(header[2:4], "little"), header[4], header[5], memory)


def verify(program: Program, result: Result, max_steps: int = 65535):
    """Compare a halted hardware result with raw CPU ISA semantics."""
    if result.fault != 0:
        raise AssertionError(f"hardware stopped with {FAULTS.get(result.fault, result.fault)}")
    vm = ISAInterpreter(32, 32, 8, list(program.memory), list(program.commands))
    for _ in range(max_steps):
        if vm.step() or vm.pc == 0:
            break
    else:
        raise AssertionError("reference interpreter reached the step limit")
    expected = (tuple(vm.memory), vm.pc, vm.carry, vm.executed)
    actual = (result.memory, result.pc, result.carry, result.steps)
    if actual != expected:
        raise AssertionError(f"FPGA/interpreter mismatch: hardware={actual}, reference={expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="FTDI channel B serial device")
    parser.add_argument("--source", type=Path, help="C or Python source file; defaults to an integer addition demo")
    parser.add_argument("--language", choices=("c", "python"), default="c")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=INTEGER")
    parser.add_argument("--max-steps", type=int, default=65535)
    parser.add_argument("--verify", action="store_true", help="compare returned memory, PC, carry, and steps with the CPU ISA interpreter")
    args = parser.parse_args()
    inputs = {}
    for item in args.input:
        name, value = item.split("=", 1)
        inputs[name] = int(value, 0)
    source = args.source.read_text() if args.source else "int x; int main() { return x + 1; }"
    if args.source is None:
        inputs.setdefault("x", 7)
    program = compile_program(source, inputs, language=args.language)
    with IceStick.open(args.port) as board:
        board.ping()
        board.load(program)
        result = board.run(args.max_steps)
    if args.verify:
        verify(program, result, args.max_steps)
    output = result.as_dict(program)
    output["verified_against_interpreter"] = args.verify
    print(json.dumps(output, indent=2))
    if result.fault:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
