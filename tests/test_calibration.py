"""Tests for confidence calibration: ECE / Brier / reliability bins."""
from calibration import (
    brier_score,
    demo_questions,
    ece,
    evaluate_calibration,
    reliability_bins,
    score_calibration,
)


def test_perfect_calibration_is_zero_ece():
    # confidence == accuracy in every bin → ECE 0.
    confs = [0.0, 0.0, 1.0, 1.0]
    correct = [0, 0, 1, 1]
    assert ece(confs, correct, n_bins=10) == 0.0


def test_overconfident_ece():
    # all 0.9 confidence, half correct → ECE = |0.5 - 0.9| = 0.4.
    confs = [0.9, 0.9, 0.9, 0.9]
    correct = [1, 1, 0, 0]
    assert ece(confs, correct, n_bins=10) == 0.4
    assert brier_score(confs, correct) == round((0.01 + 0.01 + 0.81 + 0.81) / 4, 4)


def test_reliability_bins_group_by_confidence():
    bins = reliability_bins([0.95, 0.95, 0.15], [1, 0, 0], n_bins=10)
    top = [b for b in bins if b["lo"] == 0.9][0]
    assert top["count"] == 2 and top["accuracy"] == 0.5
    assert top["mean_confidence"] == 0.95


def test_score_calibration_summary():
    out = score_calibration([0.9, 0.9, 0.9, 0.9], [1, 1, 0, 0])
    assert out["accuracy"] == 0.5
    assert out["mean_confidence"] == 0.9
    assert out["ece"] == 0.4


def test_mock_is_overconfident_end_to_end():
    # Mock answers "A" at 0.9 on all four → 2 right (gold A) / 2 wrong → ECE 0.4.
    m = evaluate_calibration(demo_questions(), model="mock")
    assert m["n"] == 4
    assert m["accuracy"] == 0.5
    assert m["mean_confidence"] == 0.9
    assert m["ece"] == 0.4
