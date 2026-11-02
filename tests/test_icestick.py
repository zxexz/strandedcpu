"""Wire format and compiler constraints for the raw iCEstick profile."""

from dataclasses import replace

import pytest

from strandedcpu.icestick import IceStick, Result, compile_program, verify
from strandedcpu.interpreter import ISAInterpreter


class Transport:
    def __init__(self, replies):
        self.replies = bytearray(replies)
        self.writes = []

    def write(self, data):
        self.writes.append(bytes(data))

    def read_exact(self, count):
        assert len(self.replies) >= count
        block = bytes(self.replies[:count])
        del self.replies[:count]
        return block


@pytest.mark.parametrize("source,language", [
    ("int x; int main() { return x + 1; }", "c"),
    ("def solution(x): return x + 1", "python"),
])
def test_compile_and_transport_agree_with_raw_interpreter(source, language):
    program = compile_program(source, {"x": 7}, language=language)
    vm = ISAInterpreter(32, 32, 8, list(program.memory), list(program.commands))
    while not vm.step():
        pass
    packet = b"D" + bytes([0]) + vm.executed.to_bytes(2, "little") + bytes([vm.pc, vm.carry])
    packet += bytes(v & 255 for v in vm.memory)
    transport = Transport(b"SC8\x01K" + packet)
    board = IceStick(transport)
    assert board.ping() == b"SC8\x01"
    board.load(program)
    result = board.run(100)
    verify(program, result, 100)
    assert result.as_dict(program)["value"] == 8
    assert result.as_dict(program)["checked"] is False
    assert transport.writes == [b"P", b"L" + program.payload(), b"Rd\x00"]


def test_rejects_unsupported_instruction_and_capacity():
    program = compile_program("int main() { return 1; }")
    with pytest.raises(ValueError, match="opcode 31"):
        replace(program, commands=((31, 32, 33),)).payload()
    with pytest.raises(ValueError, match="1..128"):
        replace(program, commands=((0, 0, 0),) * 129).payload()


def test_input_and_number_profile_is_explicit():
    with pytest.raises(ValueError, match="8-bit integer"):
        compile_program("int x; int main() { return x; }", {"x": 128})
    with pytest.raises(ValueError, match="integer programs only"):
        compile_program("fix x; int main() { return (int)x; }")
    with pytest.raises(ValueError, match="integer programs only"):
        compile_program("int main() { return (int)(1.5 * 2.0); }")
    with pytest.raises(ValueError, match="integer programs only"):
        compile_program("def solution(): return int(1.5 * 2.0)", language="python")


def test_verification_compares_state_and_detects_faults():
    program = compile_program("int main() { return 1; }")
    with pytest.raises(AssertionError, match="step_limit"):
        verify(program, Result(3, 1, 64, 0, tuple([0] * 32)))
    with pytest.raises(AssertionError, match="mismatch"):
        verify(program, Result(0, 1, 0, 0, tuple([0] * 32)))


def test_run_limit_and_firmware_validation():
    board = IceStick(Transport(b"WRNG"))
    with pytest.raises(RuntimeError, match="identifier"):
        board.ping()
    for count in (0, -1, 65536, True, 1.5):
        with pytest.raises(ValueError, match="max_steps"):
            board.run(count)


def test_jump_to_zero_halts_exactly_at_budget():
    program = compile_program("int main() { return 1; }")
    program = replace(program, commands=((3, 0, 0),))
    verify(program, Result(0, 1, 0, 0, program.memory), max_steps=1)
