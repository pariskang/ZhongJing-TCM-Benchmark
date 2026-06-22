"""Tests for the T2 patient simulator: zero leakage, inquiry loop, scoring."""
from t2_patient_sim import (
    PatientSim,
    demo_cases,
    evaluate_consultation,
    llm_expert,
    run_consultation,
    scripted_expert,
)


def _case():
    return demo_cases()[0]  # 肝郁气滞证


# -- patient simulator -------------------------------------------------------- #


def test_patient_answers_targeted_four_diagnostics():
    sim = PatientSim(_case(), model="mock")
    reply, asp = sim.answer("请问舌象如何？")
    assert asp == ["舌象"] and "舌淡红" in reply
    reply, asp = sim.answer("脉象怎么样？")
    assert asp == ["脉象"] and "弦" in reply


def test_patient_never_leaks_diagnosis():
    case = _case()
    sim = PatientSim(case, model="mock")
    for q in ["你得的是什么证？", "是不是肝郁气滞证？", "你的诊断是什么？", "病机是什么"]:
        reply, asp = sim.answer(q)
        assert asp == []
        assert case.hidden_syndrome not in reply
        assert "肝郁气滞" not in reply


def test_patient_no_leak_across_all_findings_queries():
    case = _case()
    sim = PatientSim(case, model="mock")
    for q in ["主要哪里不舒服？", "情绪如何", "睡眠怎样", "月经情况", "大便小便", "舌象", "脉象", "饮食"]:
        reply, _ = sim.answer(q)
        assert case.hidden_syndrome not in reply
        for diff in case.differentials:
            assert diff not in reply


def test_patient_redacts_label_if_present():
    case = _case()
    sim = PatientSim(case, model="mock")
    # Even if a fact text contained the label, _redact would strip it.
    assert "〔已隐去〕" in sim._redact(f"我可能是{case.hidden_syndrome}")


# -- consultation loop + scoring --------------------------------------------- #


def test_scripted_complete_consultation_is_correct():
    case = _case()
    expert = scripted_expert([
        {"action": "ask", "query": "主要不适的部位和性质如何？"},
        {"action": "ask", "query": "舌象如何？"},
        {"action": "ask", "query": "脉象如何？"},
        {"action": "diagnose", "answer": "肝郁气滞证"},
    ])
    res = run_consultation(case, expert, PatientSim(case, model="mock"))
    assert res.correct is True
    assert res.turns_used == 3
    assert res.key_feature_hits == res.key_feature_total == 3
    assert res.premature_closure is False
    assert res.abstained is False


def test_premature_closure_detected():
    # Correct answer but diagnosed immediately → process gate must flag it.
    case = _case()
    expert = scripted_expert([{"action": "diagnose", "answer": "肝郁气滞证"}])
    res = run_consultation(case, expert, PatientSim(case, model="mock"))
    assert res.correct is True            # right result …
    assert res.premature_closure is True  # … but bad process (decoupled)
    assert res.turns_used == 0
    assert res.key_feature_hits == 0


def test_diagnosis_normalisation():
    case = _case()
    # "肝郁气滞" (no 证 suffix) should still count as correct.
    expert = scripted_expert([
        {"action": "ask", "query": "部位性质？"},
        {"action": "ask", "query": "舌象？"},
        {"action": "ask", "query": "脉象？"},
        {"action": "diagnose", "answer": "肝郁气滞"},
    ])
    res = run_consultation(case, expert, PatientSim(case, model="mock"))
    assert res.correct is True


def test_abstention():
    case = _case()
    res = run_consultation(case, scripted_expert([{"action": "abstain"}]), PatientSim(case, model="mock"))
    assert res.abstained is True
    assert res.final_answer is None
    assert res.correct is False
    assert res.premature_closure is False


def test_wrong_diagnosis_is_incorrect():
    case = _case()
    expert = scripted_expert([
        {"action": "ask", "query": "舌象？"},
        {"action": "diagnose", "answer": "脾胃气虚证"},
    ])
    res = run_consultation(case, expert, PatientSim(case, model="mock"))
    assert res.correct is False


# -- mock LLM expert (end-to-end offline) ------------------------------------ #


def test_mock_llm_expert_runs_the_loop():
    case = _case()
    res = run_consultation(case, llm_expert("mock"), PatientSim(case, model="mock"), max_turns=8)
    # mock expert asks 主症 → 舌 → 脉 then diagnoses (placeholder)
    assert res.turns_used == 3
    assert res.key_feature_hits == 3
    assert res.premature_closure is False
    assert res.final_answer is not None


def test_evaluate_consultation_aggregate():
    expert = scripted_expert([
        {"action": "ask", "query": "部位性质？"},
        {"action": "ask", "query": "舌象？"},
        {"action": "ask", "query": "脉象？"},
        {"action": "diagnose", "answer": "肝郁气滞证"},
    ])
    metrics, results = evaluate_consultation([_case()], expert)
    assert metrics["n"] == 1
    assert set(metrics) >= {
        "accuracy", "mean_turns", "premature_closure_rate",
        "abstention_rate", "key_feature_recall",
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["key_feature_recall"] == 1.0
