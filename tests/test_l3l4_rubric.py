"""Tests for L3/L4: weighted rubric grading + judge meta-evaluation."""
from l3l4_rubric import (
    baseline_rubric_judge,
    demo_rubric,
    demo_samples,
    grade_response,
    meta_evaluate,
)


def test_good_response_scores_high():
    good = demo_samples()[0][0]
    score = grade_response(good, demo_rubric())
    assert score.overall == 1.0
    assert score.per_axis["safety"] == 1.0
    assert score.per_axis["accuracy"] == 1.0


def test_terse_response_scores_low():
    terse = demo_samples()[1][0]
    score = grade_response(terse, demo_rubric())
    # only the negative "contraindication" item is credited (none present) → 3/13
    assert score.overall == round(3 / 13, 4)
    assert score.per_axis["accuracy"] == 0.0


def test_negative_item_penalises_contraindication():
    rubric = demo_rubric()
    good = demo_samples()[0][0]
    base = grade_response(good, rubric).overall
    risky = good + "另加川乌、半夏同煎。"           # 十八反 contraindication present
    dropped = grade_response(risky, rubric)
    assert dropped.overall < base
    assert dropped.per_item["safe_contra"]["met"] is True
    assert dropped.per_item["safe_contra"]["credited"] is False
    assert dropped.per_axis["safety"] == 0.5        # referral kept, contraindication lost


def test_baseline_judge_met_flags():
    good = demo_samples()[0][0]
    met = baseline_rubric_judge(good, demo_rubric().items)
    assert met["acc"] is True and met["ctx"] is True
    assert met["safe_contra"] is False              # no contraindication present


def test_meta_evaluate_reports_kappa_and_concordance():
    rubric = demo_rubric()
    meta = meta_evaluate(baseline_rubric_judge, rubric, demo_samples())
    # 12 item-labels; the keyword judge disagrees with the physician on exactly one
    # (terse answer's accuracy) → concordance 11/12, kappa defined and < 1.
    assert meta["n_labels"] == 12
    assert meta["concordance"] == round(11 / 12, 4)
    assert meta["cohen_kappa"] is not None
    assert meta["cohen_kappa"] < 1.0


def test_meta_evaluate_perfect_judge():
    # A judge that exactly reproduces the physician labels → concordance 1.0.
    rubric = demo_rubric()
    samples = demo_samples()

    def oracle(response, items):
        human = next(h for r, h in samples if r == response)
        return {it.item_id: human.get(it.item_id, False) for it in items}

    meta = meta_evaluate(oracle, rubric, samples)
    assert meta["concordance"] == 1.0
