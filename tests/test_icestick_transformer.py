"""Test the SCM1 protocol and transformer arithmetic dispatch."""

import struct

import pytest
import torch

from strandedcpu.icestick_transformer import MacDevice, TransformerOffload
from strandedcpu.subleq import _attn_head
from strandedcpu.transformer import Config, Computer, OP_HALT, OP_INC, init_state, get_pc, read_memory


class ArithmeticTransport:
    """Offline SCM1 peer for host protocol tests."""

    def __init__(self):
        self.reply = bytearray()
        self.packets = []

    def write(self, data):
        self.packets.append(data)
        if data == b"P":
            self.reply.extend(b"SCM1")
            return
        assert data[0] == ord("D")
        flags, count = struct.unpack("<BH", data[1:4])
        assert len(data) == 4 + 4 * count
        accumulator, overflow = 0, 0
        for a, b in struct.iter_unpack("<hh", data[4:]):
            total = accumulator + a * b
            overflow |= not -(1 << 31) <= total < 1 << 31
            accumulator = (total + (1 << 31)) % (1 << 32) - (1 << 31)
        if flags & 1:
            accumulator = max(0, accumulator)
        self.reply.extend(b"A" + struct.pack("<iB", accumulator, overflow))

    def read_exact(self, count):
        assert len(self.reply) >= count
        answer = bytes(self.reply[:count])
        del self.reply[:count]
        return answer


def device():
    peer = ArithmeticTransport()
    return MacDevice(peer), peer


def test_protocol_signed_limits_relu_and_overflow():
    board, peer = device()
    assert board.ping() == b"SCM1"
    assert board.dot([(2, 3), (-4, 5)]) == -14
    assert board.dot([(2, 3), (-4, 5)], relu=True) == 0
    assert board.dot([(-32768, 32767)]) == -1073709056
    assert board.dot([]) == 0
    with pytest.raises(OverflowError):
        board.dot([(-32768, -32768), (-32768, -32768)])
    assert board.dot([(1, 1)]) == 1
    assert peer.packets[1] == b"D\0\2\0" + struct.pack("<hhhh", 2, 3, -4, 5)


@pytest.mark.parametrize("pairs", [[(32768, 0)], [(True, 1)], [(0.5, 2)]])
def test_protocol_rejects_unrepresentable_operands_before_send(pairs):
    board, peer = device()
    with pytest.raises(ValueError):
        board.dot(pairs)
    assert peer.packets == []


def test_firmware_identity_is_distinct_from_isa_image():
    class WrongImage:
        def write(self, _):
            pass

        def read_exact(self, _):
            return b"SC8\x01"

    with pytest.raises(RuntimeError, match="SCM1 arithmetic image"):
        MacDevice(WrongImage()).ping()


def test_exact_quarter_affine_and_hardware_relu():
    board, peer = device()
    runner = TransformerOffload(board)
    a = torch.tensor([0.25, -2.0, 0.0])
    b = torch.tensor([2.0, 3.0, 100000.0])
    actual = runner.offload_one_affine(a, b, bias=0.5, relu=True)
    assert actual == torch.relu(torch.dot(a, b) + 0.5) == 0
    assert runner.stats.offloaded_dots == runner.stats.relu_dots == 1
    assert runner.stats.offloaded_terms == 3  # two nonzero terms plus bias
    assert len(peer.packets) == 1


def test_nonrepresentable_and_unsafe_operations_use_explicit_host_fallback():
    board, peer = device()
    runner = TransformerOffload(board)
    cases = [
        (torch.tensor([924872.0]), torch.tensor([1.0]), "representation"),
        (torch.tensor([0.1]), torch.tensor([1.0]), "representation"),
        (torch.tensor([32767.0]), torch.tensor([32767.0]), "floating_exactness_bound"),
        (torch.full((3,), 32767.0), torch.full((3,), 32767.0), "accumulator_bound"),
    ]
    for a, b, reason in cases:
        assert runner.offload_one_affine(a, b) == torch.dot(a, b)
        assert runner.stats.fallback_reasons[reason] > 0
    assert runner.stats.offloaded_dots == 0
    assert peer.packets == []


def test_float64_large_product_and_budget_are_checked_separately():
    board, peer = device()
    runner = TransformerOffload(board, max_offloaded_dots=1)
    a = torch.tensor([32767.0], dtype=torch.float64)
    assert runner.offload_one_affine(a, a) == 1073676289
    assert runner.offload_one_affine(a, a) == 1073676289
    assert runner.stats.offloaded_dots == 1
    assert runner.stats.fallback_reasons == {"budget": 1}
    runner.reset_stats()
    assert runner.stats.offloaded_dots == 0
    assert runner.offload_one_affine(a, a) == 1073676289
    assert len(peer.packets) == 2


def test_nonfinite_operands_are_not_removed_by_sparse_zero_pruning():
    board, peer = device()
    runner = TransformerOffload(board)
    result = runner.offload_one_affine(torch.tensor([0.0]), torch.tensor([float("inf")]))
    assert torch.isnan(result)
    assert runner.stats.fallback_reasons == {"nonfinite": 1}
    assert peer.packets == []


def test_complete_eight_layer_step_and_halt_agree_with_pytorch():
    board, peer = device()
    cfg = Config(s=32, m=8, n=64, N=8)
    computer = Computer(cfg)
    state = init_state(cfg, [5] + [0] * 7, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    runner = TransformerOffload(board, max_offloaded_dots=8)
    with torch.no_grad():
        expected = state
        layers = []
        for layer in computer.layers:
            expected = layer.forward(expected)
            layers.append(expected)
        actual = runner.step(computer, state)
        assert len(runner.layer_outputs) == 8
        for value, baseline in zip(runner.layer_outputs, layers):
            torch.testing.assert_close(value, baseline, rtol=0, atol=1e-5)
        assert read_memory(actual, cfg)[0] == 6
        assert get_pc(actual, cfg) == 41
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
        halted = runner.step(computer, actual)
        assert get_pc(halted, cfg) == 0
        assert read_memory(halted, cfg)[0] == 6
    assert runner.stats.offloaded_dots == len(peer.packets) == 8
    assert runner.stats.offloaded_terms > 0
    assert runner.stats.fallback_reasons["budget"] > 0
    assert runner.stats.request_bytes == sum(map(len, peer.packets))
    assert runner.stats.reply_bytes == 6 * len(peer.packets)


def test_real_first_layer_affine_uses_board_relu():
    board, _ = device()
    cfg = Config(s=32, m=8, n=64, N=8)
    computer = Computer(cfg)
    state = init_state(cfg, [5] + [0] * 7, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    layer = computer.layers[0]
    attention = state + _attn_head(layer.Q, layer.K, layer.V, state, layer.lam)
    runner = TransformerOffload(board, max_offloaded_dots=1)
    actual = runner.offload_one_affine(layer.W1[1], attention[:, 0], bias=layer.b1[1, 0], relu=True)
    assert actual == torch.relu(layer.W1 @ attention + layer.b1)[1, 0] == 2
    assert runner.stats.offloaded_dots == runner.stats.relu_dots == 1


def test_zero_budget_is_host_only_and_is_reported():
    board, peer = device()
    runner = TransformerOffload(board, max_offloaded_dots=0)
    a, b = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0], [4.0]])
    assert runner.matmul(a, b).item() == 11
    assert runner.stats.fallback_reasons == {"budget": 1}
    assert peer.packets == []
