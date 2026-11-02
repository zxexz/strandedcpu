#!/usr/bin/env python3
"""Run UART ISA regressions and restore the 7+1 demo afterward.

Requires top.bin in SPI flash. Compares memory, PC, carry, and retired
instructions with ISAInterpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strandedcpu.icestick import IceStick, Program, Result, compile_program, verify
from strandedcpu.interpreter import ISAInterpreter


ADD_SOURCE = "int x; int main() { return x + 1; }"


def programs() -> list[tuple[str, Program, int, int]]:
    """(name, image, instruction limit, expected displayed value)."""
    definitions = [
        ("c_add", ADD_SOURCE, {"x": 7}, "c", 8),
        ("c_multiply", "int x; int main() { return x * 3 + 1; }", {"x": 7}, "c", 22),
        ("c_counter", "int result; int main() { result=0; while(result<10) result+=1; return result; }", {}, "c", 10),
        ("c_array_checksum", "int data[4]={1,2,3,4}; int result; int i; int main() { result=0; for(i=0;i<4;i+=1) result+=data[i]; return result; }", {}, "c", 10),
        ("c_gcd", "int a=21; int b=15; int main() { while(b!=0) { int t=b; b=a%b; a=t; } return a; }", {}, "c", 3),
        ("python_add", "def solution(x):\n    return x+1\n", {"x": 7}, "python", 8),
        ("python_square", "def solution(x):\n    return x*x\n", {"x": 7}, "python", 49),
    ]
    images = [(name, compile_program(source, inputs, language=language), 1000, expected)
              for name, source, inputs, language, expected in definitions]
    metadata = {"variables": {}, "sizes": {}, "retval_addr": 0}
    boundary = Program(tuple([0]*32), tuple([(6, 32, 0)]*127+[(0, 0, 0)]), metadata)
    images.append(("raw_128_instruction_boundary", boundary, 1000, 127))
    return images


def entry(name: str, program: Program, result: Result) -> dict:
    record = result.as_dict(program)
    return {"name": name, "instructions": len(program.commands),
            "value": record["value"], "status": record["status"],
            "steps": result.steps, "pc": result.pc, "carry": result.carry,
            "memory": list(result.memory), "full_state_matches_interpreter": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent/"build"/"isa-on-device.json")
    args = parser.parse_args()
    images = programs()
    initial_demo = compile_program(ADD_SOURCE, {"x": 7})
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "port": args.port, "backend": "icestick", "profile": "raw-int8",
              "checked": False, "cases": [], "errors": []}
    try:
        with IceStick.open(args.port) as board:
            report["firmware_id_hex"] = board.ping().hex()
            try:
                for name, image, limit, expected in images:
                    board.load(image)
                    result = board.run(limit)
                    verify(image, result, limit)
                    record = entry(name, image, result)
                    if record["value"] != expected:
                        raise AssertionError(f"{name}: expected {expected}, received {record['value']}")
                    report["cases"].append(record)
                    print(f"PASS {name}: value={record['value']}, steps={result.steps}", flush=True)

                # Step-limit termination is a profile fault rather than HALT;
                # compare the exact state after the same instruction budget.
                image = Program(tuple([0]*32), ((3, 0, 64),),
                                {"variables": {}, "sizes": {}, "retval_addr": 0})
                limit = 257
                board.load(image)
                result = board.run(limit)
                reference = ISAInterpreter(32, 32, 8, list(image.memory), list(image.commands))
                for _ in range(limit):
                    if reference.step():
                        raise AssertionError("step-limit reference unexpectedly halted")
                expected = Result(3, reference.executed, reference.pc, reference.carry,
                                  tuple(reference.memory))
                if result != expected:
                    raise AssertionError(f"step-limit state mismatch: {result!r} != {expected!r}")
                report["cases"].append(entry("raw_step_limit_257", image, result))
                print(f"PASS raw_step_limit_257: status=step_limit, steps={result.steps}", flush=True)
            finally:
                try:
                    board.load(initial_demo)
                    restored = board.run(1000)
                    verify(initial_demo, restored, 1000)
                    report["restored_demo"] = entry("c_add", initial_demo, restored)
                    if report["restored_demo"]["value"] != 8:
                        raise AssertionError("restored demo did not return 8")
                    print("Restored addition demo: 7+1=8", flush=True)
                except Exception as exc:
                    report["errors"].append(f"restore: {type(exc).__name__}: {exc}")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "pass" if not report["errors"] and len(report["cases"]) == len(images)+1 else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Serial port closed; report: {args.output}", flush=True)
    if report["status"] != "pass":
        raise SystemExit("; ".join(report["errors"]) or "incomplete board regression")


if __name__ == "__main__":
    main()
