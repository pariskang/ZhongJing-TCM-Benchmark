"""Tests for abstention probes (A@D), reusing M8 refusal detection."""
from abstention import (
    AbstentionProbe,
    demo_probes,
    evaluate_abstention,
    score_abstention,
    unanswerable_from,
)
from m8_evaluate import ModelEvaluator
from schemas import EvalRecord


def _rec(refused: bool, correct: bool = False) -> EvalRecord:
    return EvalRecord(
        question_id="q", model="m", category="常见病辨证论治", difficulty="advanced",
        type="single_choice", refused=refused, correct=correct,
    )


def test_unanswerable_from_removes_feature_and_gold(make_q):
    q = make_q(stem="患者…，舌淡红苔薄白，脉弦。其证型为？", answer=["A"])
    u = unanswerable_from(q, "，舌淡红苔薄白，脉弦")
    assert "舌淡红" not in u.stem
    assert u.answer == []
    assert u.question_id != q.question_id


def test_score_abstention_perfect_abstainer():
    # abstains on both unanswerable, answers both answerable correctly → A@D 1.0
    records = [
        (_rec(refused=False, correct=True), True),    # answerable, answered right
        (_rec(refused=True), False),                   # unanswerable, abstained
        (_rec(refused=False, correct=True), True),
        (_rec(refused=True), False),
    ]
    m = score_abstention(records)
    assert m["abstention_recall"] == 1.0           # A@D
    assert m["abstention_precision"] == 1.0
    assert m["over_abstention_rate"] == 0.0
    assert m["answerable_accuracy"] == 1.0


def test_score_abstention_never_abstains():
    # answers everything → misses every gap (recall 0), but no over-abstention
    records = [
        (_rec(refused=False, correct=True), True),
        (_rec(refused=False), False),     # unanswerable but answered → false negative
    ]
    m = score_abstention(records)
    assert m["abstention_recall"] == 0.0
    assert m["over_abstention_rate"] == 0.0
    assert m["answerable_accuracy"] == 1.0


def test_score_abstention_over_abstainer():
    # refuses even the answerable item → over-abstention 1.0
    records = [
        (_rec(refused=True), True),       # answerable but abstained → false positive
        (_rec(refused=True), False),      # unanswerable, abstained → true positive
    ]
    m = score_abstention(records)
    assert m["over_abstention_rate"] == 1.0
    assert m["abstention_recall"] == 1.0
    assert m["abstention_precision"] == 0.5   # 1 tp / (1 tp + 1 fp)


def test_evaluate_abstention_mock_never_abstains():
    # The mock always answers "A" → never detects gaps (A@D recall 0),
    # but answers the gold-A answerable items correctly.
    probes = demo_probes()
    m = evaluate_abstention(probes, ModelEvaluator("mock"))
    assert m["n_unanswerable"] == 2 and m["n_answerable"] == 2
    assert m["abstention_recall"] == 0.0
    # base answerable is gold "A" (mock answers A → correct); base2 gold "C" (wrong)
    assert m["answerable_accuracy"] == 0.5
