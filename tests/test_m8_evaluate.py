"""Tests for M8 evaluation: refusal parsing, answer parsing, scoring."""
from m8_evaluate import ModelEvaluator


def _ev():
    return ModelEvaluator("mock")


def test_parse_explicit_refusal():
    out = _ev().parse("根据题目信息不足，无法确定正确答案。")
    assert out["refused"] is True
    assert out["pred"] == []


def test_parse_implicit_refusal_no_answer():
    out = _ev().parse("这是一段没有给出任何选项的分析。")
    assert out["refused"] is True  # no option selected → implicit refusal


def test_parse_single_answer():
    out = _ev().parse("1. 答案选择\n   - [Answer] A\n2. 详细分析 ...")
    assert out == {"refused": False, "pred": ["A"]}


def test_parse_multi_answer():
    out = _ev().parse("[Answer] A、C")
    assert out["pred"] == ["A", "C"]


def test_eval_one_correct(make_q):
    ev = _ev()
    rec = ev.eval_one(make_q(answer=["A"]))   # mock STAGER answers "A"
    assert rec.pred == ["A"]
    assert rec.correct is True
    assert rec.refused is False
    assert rec.output_tokens > 0


def test_evaluate_metrics(make_q):
    dataset = [make_q(answer=["A"]) for _ in range(3)] + [make_q(answer=["A", "C"]) for _ in range(3)]
    metrics, rows = _ev().evaluate(dataset)
    assert metrics["n"] == 6
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] == 0.5  # 3 single (correct) / 3 multi (wrong)
    assert metrics["refusal_rate"] == 0.0
