"""Tests for T1: counterfactual minimal pairs + information staging."""
import uuid

from m8_evaluate import ModelEvaluator
from schemas import Category, Passage
from t1_counterfactual import (
    StagedCase,
    evaluate_staging,
    generate_counterfactual,
    score_counterfactual_pairs,
    staged_from_question,
)


def _passage():
    return Passage(
        passage_id=str(uuid.uuid4()), article_id="a",
        text="脾胃虚寒与脾胃湿热的鉴别，关键在舌脉。",
        topic_id=0, category=Category.COMMON_DISEASE, topic_keywords=["脾胃"],
    )


# -- counterfactual pairs ----------------------------------------------------- #


def test_generate_counterfactual_pair_flips_answer():
    pair = generate_counterfactual(_passage(), model="mock")
    assert pair is not None and pair.is_valid()
    assert pair.base.answer != pair.variant.answer          # answer flips
    assert pair.base.options == pair.variant.options        # options shared
    # the two stems differ only by the feature value
    assert pair.base_value not in pair.variant.stem
    assert pair.cf_value in pair.variant.stem


def test_score_counterfactual_pairs_catches_position_bias():
    # Mock always answers the first label "A": base (A) right, variant (B) wrong
    # → pair_accuracy 0, exposing that the model ignores the flipped feature.
    pair = generate_counterfactual(_passage(), model="mock")
    report = score_counterfactual_pairs([pair], ModelEvaluator("mock"))
    assert report["n"] == 1
    assert report["base_accuracy"] == 1.0
    assert report["variant_accuracy"] == 0.0
    assert report["pair_accuracy"] == 0.0   # cannot get both → feature not used
    assert report["flip_rate"] == 0.0


# -- information staging ------------------------------------------------------ #


def test_staged_from_question_orders_units(make_q):
    q = make_q(
        stem="患者女，35岁，胸胁胀痛，善太息，舌淡红苔薄白，脉弦。其证型为？",
        answer=["A"],
    )
    case = staged_from_question(q, n_stages=4)
    assert 1 <= len(case.stages) <= 4
    assert case.options == q.options and case.answer == q.answer
    # stages are non-empty and reconstruct (loosely) the stem content
    assert all(case.stages)


def test_evaluate_staging_finds_min_correct_stage(make_q):
    # Mock answers "A"; gold ["A"] → correct from the first stage already.
    q = make_q(stem="患者反复发热。继而恶寒。舌红苔黄。脉数有力。", answer=["A"])
    case = staged_from_question(q, n_stages=4)
    out = evaluate_staging(case, ModelEvaluator("mock"))
    assert out["min_correct_stage"] == 1
    assert out["information_efficiency"] == 1.0
    assert out["early_wrong"] is False


def test_evaluate_staging_never_correct(make_q):
    # Gold is "C" but the mock always answers "A" → never correct.
    q = make_q(stem="患者反复发热。继而恶寒。舌红苔黄。脉数有力。", answer=["C"])
    case = staged_from_question(q, n_stages=4)
    out = evaluate_staging(case, ModelEvaluator("mock"))
    assert out["min_correct_stage"] is None
    assert out["information_efficiency"] == 0.0
