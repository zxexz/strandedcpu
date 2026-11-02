"""Offload selected transformer dot products to the iCEstick SCM1 image.

The board evaluates bounded integer dot products and optional ReLU. The host
retains weights, state, attention selection, residuals and scheduling, and
uses PyTorch for operations outside the representation or UART dot budget.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
import struct

import torch

from strandedcpu.icestick import SerialTransport
from strandedcpu.subleq import _top2_weights


class MacDevice:
    """UART client for the SCM1 dot-product firmware."""

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
        if reply != b"SCM1":
            raise RuntimeError(f"expected SCM1 arithmetic image, received {reply!r}")
        return reply

    def dot(self, pairs, *, relu: bool = False) -> int:
        pairs = list(pairs)
        if len(pairs) > 65535:
            raise ValueError("SCM1 supports at most 65535 dot terms")
        data = bytearray(b"D" + struct.pack("<BH", int(bool(relu)), len(pairs)))
        for pair in pairs:
            if len(pair) != 2 or any(type(v) is not int or not -32768 <= v <= 32767 for v in pair):
                raise ValueError("dot operands must be signed 16-bit integers")
            data.extend(struct.pack("<hh", *pair))
        self.transport.write(bytes(data))
        reply = self.transport.read_exact(6)
        if reply[0] != ord("A") or reply[5] not in (0, 1):
            raise RuntimeError(f"invalid SCM1 dot reply: {reply!r}")
        if reply[5]:
            raise OverflowError("SCM1 accumulator overflowed; result is invalid")
        return struct.unpack("<i", reply[1:5])[0]


@dataclass
class OffloadStats:
    offloaded_dots: int = 0
    offloaded_terms: int = 0
    relu_dots: int = 0
    zero_dots: int = 0
    fallback_dots: int = 0
    fallback_reasons: Counter = field(default_factory=Counter)
    request_bytes: int = 0
    reply_bytes: int = 0

    def as_dict(self):
        return {**self.__dict__, "fallback_reasons": dict(self.fallback_reasons)}


def _scaled_int16(values):
    """Return exact int16 operands and their fractional-bit count, or None."""
    values = values.detach().double()
    if not bool(torch.isfinite(values).all()):
        return None
    for bits in range(15):
        scaled = values * (1 << bits)
        if bool(((scaled >= -32768) & (scaled <= 32767) & (scaled == scaled.round())).all()):
            return [int(v) for v in scaled.tolist()], bits
    return None


class TransformerOffload:
    """Evaluate transformer matrix operations with a cumulative UART dot budget.

    Send a dot to the board only when its scaled operands fit int16 and its
    products and partial sums fit both the accumulator and source float precision.
    Otherwise use PyTorch and count the fallback reason. reset_stats() also resets
    the dot budget.
    """

    def __init__(self, board: MacDevice, *, max_offloaded_dots: int | None = None):
        if max_offloaded_dots is not None and (type(max_offloaded_dots) is not int or max_offloaded_dots < 0):
            raise ValueError("max_offloaded_dots must be a nonnegative integer or None")
        self.board = board
        self.max_offloaded_dots = max_offloaded_dots
        self.reset_stats()

    def reset_stats(self):
        self.stats = OffloadStats()
        self.layer_outputs = []

    def _budget_exhausted(self):
        return self.max_offloaded_dots is not None and self.stats.offloaded_dots >= self.max_offloaded_dots

    def _fallback(self, reason: str, count: int = 1):
        self.stats.fallback_dots += count
        self.stats.fallback_reasons[reason] += count

    def _prepare(self, coefficients, activations, bias):
        if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(activations).all()):
            return None, None, "nonfinite"
        # Sparse traversal removes only mathematically zero products.
        live = (coefficients != 0) & (activations != 0)
        a, b = coefficients[live], activations[live]
        if not len(a) and float(bias) == 0:
            return [], 0, None
        a_scaled, b_scaled = _scaled_int16(a), _scaled_int16(b)
        if a_scaled is None or b_scaled is None:
            return None, None, "representation"
        ai, af = a_scaled
        bi, bf = b_scaled
        bits = af + bf
        bias_scaled = float(bias) * (1 << bits)
        if not math.isfinite(bias_scaled) or not bias_scaled.is_integer() or not -32768 <= bias_scaled <= 32767:
            return None, None, "bias_representation"
        pairs = list(zip(ai, bi))
        if bias_scaled:
            pairs.append((int(bias_scaled), 1))
        if len(pairs) > 65535:
            return None, None, "term_limit"
        bound = sum(abs(x) * abs(y) for x, y in pairs)
        if bound > (1 << 31) - 1:
            return None, None, "accumulator_bound"
        precision = 53 if coefficients.dtype == torch.float64 else 24
        # Bounding the sum of absolute products also bounds every partial sum,
        # preserving float exactness regardless of summation order.
        if bound > 1 << precision:
            return None, None, "floating_exactness_bound"
        return pairs, bits, None

    def offload_one_affine(self, coefficients, activations, *, bias=0.0, relu=False):
        """Evaluate one affine dot, with hardware eligibility and fallback recorded in stats."""
        if coefficients.ndim != 1 or activations.shape != coefficients.shape:
            raise ValueError("affine operands must be equal-length vectors")
        if coefficients.device.type != "cpu" or activations.device.type != "cpu":
            raise ValueError("the SCM1 prototype accepts CPU tensors")
        if coefficients.dtype not in (torch.float32, torch.float64) or activations.dtype != coefficients.dtype:
            raise ValueError("operands must share float32 or float64 dtype")
        if self._budget_exhausted():
            self._fallback("budget")
        else:
            pairs, bits, reason = self._prepare(coefficients, activations, bias)
            if reason is None:
                if not pairs:
                    self.stats.zero_dots += 1
                    return coefficients.new_tensor(0.0)
                value = self.board.dot(pairs, relu=relu)
                self.stats.offloaded_dots += 1
                self.stats.offloaded_terms += len(pairs)
                self.stats.relu_dots += int(bool(relu))
                self.stats.request_bytes += 4 + 4 * len(pairs)
                self.stats.reply_bytes += 6
                return coefficients.new_tensor(math.ldexp(float(value), -bits))
            self._fallback(reason)
        value = torch.dot(coefficients, activations) + bias
        return torch.relu(value) if relu else value

    def matmul(self, left, right, *, bias=None, relu=False):
        """Matrix product with hardware dots and batched host fallback blocks."""
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError("matrix operands must be compatible rank-two tensors")
        if (left.device.type != "cpu" or right.device.type != "cpu" or left.dtype != right.dtype
                or left.dtype not in (torch.float32, torch.float64)):
            raise ValueError("matrix operands must share a CPU floating dtype")
        rows, cols = left.shape[0], right.shape[1]
        if bias is not None and bias.shape != (rows, cols):
            raise ValueError("bias must have the output matrix's shape")
        out = left.new_empty((rows, cols))
        for row in range(rows):
            if self._budget_exhausted():
                value = left[row:] @ right
                if bias is not None:
                    value = value + bias[row:]
                out[row:] = torch.relu(value) if relu else value
                self._fallback("budget", (rows - row) * cols)
                break
            for col in range(cols):
                if self._budget_exhausted():
                    value = left[row:row + 1] @ right[:, col:]
                    if bias is not None:
                        value = value + bias[row:row + 1, col:]
                    out[row:row + 1, col:] = torch.relu(value) if relu else value
                    self._fallback("budget", cols - col)
                    break
                out[row, col] = self.offload_one_affine(left[row], right[:, col],
                    bias=0.0 if bias is None else bias[row, col], relu=relu)
        return out

    def _head(self, layer, head, state):
        suffix = "" if layer.num_heads == 1 else str(head)
        q, k, v = (getattr(layer, name + suffix) for name in ("Q", "K", "V"))
        # These heads are structurally absent, so no arithmetic is required.
        if not bool(torch.count_nonzero(q)) and not bool(torch.count_nonzero(k)):
            return torch.zeros_like(state)
        qx, kx, vx = (self.matmul(matrix, state) for matrix in (q, k, v))
        if bool(qx.abs().max() < 1e-6) and bool(kx.abs().max() < 1e-6):
            return torch.zeros_like(state)
        scores = self.matmul(kx.T.double(), qx.double())
        if head in layer.query_wise:
            live_key = kx.abs().amax(dim=0) > 1e-6
            live_query = qx.abs().amax(dim=0) > 1e-6
            scores = scores.masked_fill(~live_key.unsqueeze(1), float("-inf"))
            vals, indices = scores.topk(2, dim=0)
            active = live_query & torch.isfinite(vals[0])
            tied = active & torch.isfinite(vals[1]) & (vals[0] - vals[1] < 1.0)
            weights = torch.zeros_like(scores, dtype=state.dtype)
            weights.scatter_(0, indices[:1], (torch.where(tied, 0.5, 1.0) * active).to(state.dtype).unsqueeze(0))
            weights.scatter_add_(0, indices[1:], (torch.where(tied, 0.5, 0.0)).to(state.dtype).unsqueeze(0))
        else:
            weights = _top2_weights(scores, kx.abs().amax(dim=0) >= 1e-6)
        return self.matmul(vx, weights)

    def step(self, computer, state):
        """Execute the computer's layers, sharing this object's remaining dot budget."""
        if state.device.type != "cpu" or state.dtype != torch.float32:
            raise ValueError("step requires a CPU float32 transformer state")
        if tuple(state.shape) != (computer.cfg.d_model, computer.cfg.n):
            raise ValueError("state shape does not match the transformer")
        self.layer_outputs = []
        with torch.no_grad():
            for layer in computer.layers:
                attention = state
                for head in range(1, layer.num_heads + 1):
                    attention = attention + self._head(layer, head, state)
                hidden = self.matmul(layer.W1, attention, bias=layer.b1, relu=True)
                state = attention + self.matmul(layer.W2, hidden) + layer.b2
                self.layer_outputs.append(state.clone())
        return state


def main():
    from strandedcpu.transformer import Config, Computer, OP_INC, OP_HALT, init_state, get_pc, read_memory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--max-offloaded-dots", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    cfg = Config(s=32, m=8, n=64, N=8)
    computer = Computer(cfg)
    state = init_state(cfg, [5] + [0] * 7, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
    max_error = 0.0
    executed = 0
    with MacDevice.open(args.port) as board, torch.no_grad():
        board.ping()
        runner = TransformerOffload(board, max_offloaded_dots=args.max_offloaded_dots)
        for _ in range(args.steps):
            expected = computer.step(state) if args.verify else None
            state = runner.step(computer, state)
            executed += 1
            if expected is not None:
                max_error = max(max_error, float((state - expected).abs().max()))
                torch.testing.assert_close(state, expected, rtol=0, atol=1e-5)
                if (read_memory(state, cfg), get_pc(state, cfg)) != (read_memory(expected, cfg), get_pc(expected, cfg)):
                    raise AssertionError("decoded transformer state disagrees with PyTorch")
            if get_pc(state, cfg) == 0:
                break
    print(json.dumps({"backend": "icestick-transformer-hybrid", "device_resident": False,
        "steps": executed, "pc": get_pc(state, cfg), "memory": read_memory(state, cfg),
        "verified_against_pytorch": args.verify, "max_state_error": max_error if args.verify else None,
        "offload": runner.stats.as_dict()}, indent=2))


if __name__ == "__main__":
    main()
