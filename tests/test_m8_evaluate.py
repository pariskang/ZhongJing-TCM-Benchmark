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
    assert "short_answer_accuracy" not in metrics  # no SA questions in this batch


def test_eval_one_short_answer(make_q):
    ev = _ev()
    sa_q = make_q(type="short_answer", reference_answer="柴胡，疏解少阳，透邪外出。")
    rec = ev.eval_one_short_answer(sa_q)
    assert rec is not None
    assert rec.type == "short_answer"
    assert rec.correct is True   # mock judge always returns correct=True
    assert rec.output_tokens > 0


def test_evaluate_includes_short_answer(make_q):
    sa = make_q(type="short_answer", reference_answer="柴胡，疏解少阳，透邪外出。")
    choice = make_q(answer=["A"])
    metrics, rows = _ev().evaluate([sa, choice])
    assert metrics["n"] == 2
    assert "short_answer_accuracy" in metrics
    assert metrics["short_answer_accuracy"] == 1.0  # mock judge correct=True
    sa_recs = [r for r in rows if r.type == "short_answer"]
    assert len(sa_recs) == 1


def test_eval_short_answer_without_reference(make_q):
    ev = _ev()
    sa_q = make_q(type="short_answer")
    sa_q.reference_answer = None  # override the fixture default
    # No reference_answer → eval_one_short_answer returns None (skipped).
    rec = ev.eval_one_short_answer(sa_q)
    assert rec is None
