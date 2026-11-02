"""Test question alignment and answer labels in the model adapter."""

import pytest

from strandedcpu.gemma import check_programs, decisions, extract_program, summarize


def test_extract_last_program():
    text = "```python\ndef solution(): return 1\n```\n```python\ndef solution(): return 2\n```"
    assert extract_program(text) == "def solution(): return 2\n"
    assert extract_program("no program") is None


def test_check_programs_exact_and_missing():
    results = check_programs(
        ["def solution(): return 50 * 0.8", None], ["40", "0"], backend="interpreter",
    )
    assert results[0]["status"] == "ok"
    assert results[1]["abstain_reason"] == "compile_error"
    assert summarize(results) == {
        "n": 2, "correct": 1, "accuracy": 0.5, "failures": {"compile_error": 1},
    }


def test_no_numeric_options_cannot_invent_a_selection():
    assert decisions(
        ["def solution(): return 0"], [["A) None of these", "B) Unknown"]],
        backend="interpreter",
    ) == [None]


def test_decisions_preserve_question_order_when_skipping_options():
    assert decisions(
        ["def solution(): return 40", "def solution(): return 0", "def solution(): return 41"],
        [["A) $40", "B) $41"], ["A) None of these"], ["A) $40", "B) $41"]],
        backend="interpreter",
    ) == ["A", None, "B"]


@pytest.mark.parametrize("function", [check_programs, decisions])
def test_mismatched_questions_are_rejected(function):
    with pytest.raises(ValueError, match="equal lengths"):
        function(["def solution(): return 1"], [], backend="interpreter")
