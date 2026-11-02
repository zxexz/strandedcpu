"""Compare SCM1 arithmetic and hybrid transformer steps with PyTorch.

Requires mac.bin in SPI flash. Reports per-layer errors and FPGA/host dot counts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Allow direct execution from an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from strandedcpu.icestick_transformer import MacDevice, TransformerOffload
from strandedcpu.subleq import _attn_head
from strandedcpu.transformer import Config, Computer, OP_INC, OP_HALT, init_state, get_pc, read_memory


def validate(port: str, *, dot_budget: int = 16, timeout: float = 5.0) -> dict:
    report = {
        "status": "running",
        "validation": "physical-board",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "port": port,
        "backend": "icestick-transformer-hybrid",
        "device_resident": False,
        "host_operations": ["state and weights", "layer scheduling", "attention selection",
                            "residual additions", "unrepresentable and over-budget arithmetic"],
        "dot_budget": dot_budget,
        "arithmetic_cases": [],
    }
    with MacDevice.open(port, timeout=timeout) as board, torch.no_grad():
        report["firmware"] = board.ping().decode("ascii")
        cases = [
            ("zero_terms", [], False, 0),
            ("positive_limit_product", [(32767, 32767)], False, 1073676289),
            ("minimum_signed_squared", [(-32768, -32768)], False, 1073741824),
            ("minimum_signed_mixed_sign", [(-32768, 32767)], False, -1073709056),
            ("signed_multi_term", [(300, -7), (-12, -25), (0, -32768)], False, -1800),
            ("completed_dot_relu", [(2, 3), (-4, 5)], True, 0),
        ]
        for name, pairs, relu, expected in cases:
            actual = board.dot(pairs, relu=relu)
            if actual != expected:
                raise AssertionError(f"{name}: board returned {actual}, expected {expected}")
            report["arithmetic_cases"].append({"case": name, "value": actual, "passed": True})

        for name, pairs in [
            ("positive_accumulator_overflow", [(-32768, -32768)] * 2),
            ("negative_accumulator_overflow", [(-32768, 32767)] * 3),
        ]:
            try:
                board.dot(pairs)
            except OverflowError:
                pass
            else:
                raise AssertionError(f"{name}: board did not report overflow")
            recovered = board.dot([(1, 7)])
            if recovered != 7:
                raise AssertionError(f"{name}: next request failed to clear accumulator/overflow")
            report["arithmetic_cases"].append({"case": name, "overflow_reported": True,
                "next_request_value": recovered, "passed": True})

        cfg = Config(s=32, m=8, n=64, N=8)
        computer = Computer(cfg)
        state = init_state(cfg, [5] + [0] * 7, [(OP_INC, cfg.s, 0), (OP_HALT, 0, 0)])
        report["configuration"] = {"s": cfg.s, "m": cfg.m, "n": cfg.n, "N": cfg.N,
            "d_model": cfg.d_model, "layers": len(computer.layers),
            "program": [[OP_INC, cfg.s, 0], [OP_HALT, 0, 0]]}

        # First-layer neuron after attention, including bias.
        layer = computer.layers[0]
        attention = state + _attn_head(layer.Q, layer.K, layer.V, state, layer.lam)
        affine = TransformerOffload(board, max_offloaded_dots=1)
        actual_neuron = affine.offload_one_affine(layer.W1[1], attention[:, 0],
            bias=layer.b1[1, 0], relu=True)
        expected_neuron = torch.relu(layer.W1 @ attention + layer.b1)[1, 0]
        if actual_neuron.item() != 2 or actual_neuron.item() != expected_neuron.item():
            raise AssertionError("real first-layer affine/ReLU disagrees with PyTorch")
        if affine.stats.offloaded_dots != 1 or affine.stats.relu_dots != 1:
            raise AssertionError("affine/ReLU validation did not execute on the board")
        report["real_layer1_neuron"] = {"row": 1, "column": 0,
            "value": actual_neuron.item(), "reference": expected_neuron.item(),
            "offload": affine.stats.as_dict(), "passed": True}

        runner = TransformerOffload(board, max_offloaded_dots=dot_budget)
        step_reports = []
        for step in range(1, 3):
            expected, reference_layers = state, []
            for layer in computer.layers:
                expected = layer.forward(expected)
                reference_layers.append(expected)
            actual = runner.step(computer, state)
            layer_errors = []
            for number, (value, baseline) in enumerate(zip(runner.layer_outputs, reference_layers), 1):
                error = float((value - baseline).abs().max())
                torch.testing.assert_close(value, baseline, rtol=0, atol=1e-5,
                    msg=lambda text, number=number: f"step {step}, layer {number}: {text}")
                layer_errors.append(error)
            if len(layer_errors) != 8:
                raise AssertionError("validation did not execute all eight transformer layers")
            memory, pc = read_memory(actual, cfg), get_pc(actual, cfg)
            carry = int(actual[cfg.idx_carry, 0].item() > 0.5)
            expected_carry = int(expected[cfg.idx_carry, 0].item() > 0.5)
            if (memory, pc, carry) != (read_memory(expected, cfg), get_pc(expected, cfg), expected_carry):
                raise AssertionError("decoded memory, PC, or carry disagrees with PyTorch")
            step_reports.append({"step": step, "layer_max_abs_errors": layer_errors,
                "memory": memory, "pc": pc, "carry": carry, "passed": True})
            state = actual
        if runner.stats.offloaded_dots <= 0 or runner.stats.offloaded_terms <= 0:
            raise AssertionError("complete-step validation performed no FPGA arithmetic")
        if runner.stats.fallback_dots <= 0:
            raise AssertionError("prototype coverage must explicitly report host fallback")
        if read_memory(state, cfg)[0] != 6 or get_pc(state, cfg) != 0:
            raise AssertionError("INC/HALT program did not produce memory[0]=6 and halt")
        report["transformer_steps"] = step_reports
        report["step_offload"] = runner.stats.as_dict()
        report["coverage"] = "Limited dot budget; all layers execute, remaining arithmetic uses host fallback"
        report["status"] = "passed"
    report["serial_closed"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="FTDI channel B serial device")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-offloaded-dots", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.max_offloaded_dots < 1:
        parser.error("--max-offloaded-dots must be positive for physical arithmetic validation")
    report = validate(args.port, dot_budget=args.max_offloaded_dots, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "firmware": report["firmware"],
        "real_layer1_relu": report["real_layer1_neuron"]["value"],
        "hardware_dots": report["step_offload"]["offloaded_dots"],
        "host_fallback_dots": report["step_offload"]["fallback_dots"],
        "final_pc": report["transformer_steps"][-1]["pc"],
        "final_memory0": report["transformer_steps"][-1]["memory"][0],
        "max_layer_error": max(max(step["layer_max_abs_errors"]) for step in report["transformer_steps"]),
        "serial_closed": report["serial_closed"], "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
