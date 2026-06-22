"""Tests for M8 evaluation: refusal parsing, answer parsing, scoring."""
from m8_evaluate import ModelEvaluator, relabel_options, shuffle_options


def _ev():
    return ModelEvaluator("mock")


# -- option-order / symbol invariance ---------------------------------------- #


def test_shuffle_options_remaps_gold(make_q):
    q = make_q(answer=["A"])  # A = 柴胡
    gold_content = q.options[q.answer[0]]
    sq = shuffle_options(q, seed=3)
    assert sorted(sq.options.values()) == sorted(q.options.values())  # same contents
    assert sq.options[sq.answer[0]] == gold_content                   # gold follows content


def test_relabel_options_cjk(make_q):
    q = make_q(answer=["A"])
    rq = relabel_options(q, "cjk")
    assert set(rq.options.keys()) == {"甲", "乙", "丙", "丁"}
    assert rq.answer == ["甲"]
    assert rq.options["甲"] == q.options["A"]


def test_parse_cjk_labels():
    out = _ev().parse("1. 答案选择\n   - [Answer] 甲", labels=("甲", "乙", "丙", "丁"))
    assert out == {"refused": False, "pred": ["甲"]}


def test_evaluate_invariance_structure_and_shuffle_bias(make_q):
    # The mock always picks the first label "A" (position bias).
    q = make_q(answer=["A"])
    q.question_id = "inv-fixed"               # deterministic shuffle permutation
    rep = _ev().evaluate_invariance([q], perturbations=("shuffle", "cjk"))
    assert rep["base_accuracy"] == 1.0
    assert set(rep["perturbations"]) == {"shuffle", "cjk"}
    # relabel preserves position → the position-biased mock stays correct
    assert rep["perturbations"]["cjk"]["consistency"] == 1.0
    assert rep["perturbations"]["cjk"]["accuracy_drop"] == 0.0
    # shuffle: correct iff the gold content happens to remain under key "A"
    sq = shuffle_options(q, 0)
    expected_drop = 0.0 if sq.options["A"] == q.options["A"] else 1.0
    assert rep["perturbations"]["shuffle"]["accuracy_drop"] == expected_drop
    assert rep["perturbations"]["shuffle"]["consistency"] == 1.0 - expected_drop


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
