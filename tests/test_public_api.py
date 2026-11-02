"""Test package entry points for checked C and Python execution."""

import strandedcpu


def test_public_c_and_python_entry_points():
    result = strandedcpu.run_function(
        "int x; int main() { return x * x; }", {"x": 7}
    )
    assert result["__retval"] == 49

    result = strandedcpu.score_python(
        "def solution():\n    return 0.1 + 0.2\n",
        {}, {"A": "0.3", "B": "0.4"}, N=32, numbers="rational",
    )
    assert result["status"] == "ok"
    assert result["value"] == 0.3
    assert result["value_raw"] == (3, 10)
    assert result["match"] == {"A": True, "B": False}
    assert result["strandedcpu_commit"]
    assert "strandedcpu_diff_sha256" in result
    assert not any(key.startswith("loom_") for key in result)


def test_public_checked_execution_abstains_on_overflow():
    result = strandedcpu.score_options(
        "int main() { return 100 * 100; }", {}, {"A": "10000"}, N=8
    )
    assert result["status"] == "abstain"
    assert result["abstain_reason"] == "overflow"
