"""Tests for L2: step-PRM data, process preference, and the process gate."""
from l2_process import (
    baseline_process_judge,
    build_step_cases,
    gate_consultations,
    llm_process_judge,
    process_gate,
    score_process,
)
from t2_patient_sim import ConsultationResult, demo_cases


def _steps():
    return [sc for c in demo_cases() for sc in build_step_cases(c)]


def test_build_step_cases_distinct_actions():
    steps = _steps()
    assert len(steps) == 4  # 2 cases × 2 decision points
    for sc in steps:
        assert sc.correct_action and sc.wrong_action
        assert sc.correct_action != sc.wrong_action
        assert sc.neutral_action is not None


def test_baseline_judge_prefers_correct():
    rep = score_process(_steps(), baseline_process_judge)
    assert rep["process_preference_accuracy"] == 1.0
    assert rep["correct_vs_neutral_accuracy"] == 1.0


def test_inverted_judge_scores_zero():
    inverted = lambda c, a, b: "B" if baseline_process_judge(c, a, b) == "A" else "A"
    rep = score_process(_steps(), inverted)
    assert rep["process_preference_accuracy"] == 0.0


def test_mock_llm_process_judge_prefers_correct():
    rep = score_process(_steps(), llm_process_judge("mock"))
    assert rep["process_preference_accuracy"] == 1.0


def test_process_gate_decouples_result_and_process():
    assert process_gate(True, True)["gated_pass"] is True
    assert process_gate(True, False)["gated_pass"] is False   # right answer, bad process
    assert process_gate(False, True)["gated_pass"] is False


def _result(correct, premature):
    return ConsultationResult(
        case_id="c", model="m", turns_used=0, final_answer="x",
        correct=correct, premature_closure=premature,
        key_feature_hits=0, key_feature_total=3,
    )


def test_gate_consultations_downgrades_premature_correct():
    results = [
        _result(correct=True, premature=False),   # kept
        _result(correct=True, premature=True),    # downgraded
        _result(correct=False, premature=False),  # wrong anyway
    ]
    gate = gate_consultations(results)
    assert gate["raw_accuracy"] == round(2 / 3, 4)
    assert gate["gated_accuracy"] == round(1 / 3, 4)
    assert gate["downgraded"] == 1
