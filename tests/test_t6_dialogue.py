"""Tests for T6 open-ended multi-turn rubric dialogue."""
from l3l4_rubric import Rubric, RubricItem
from t6_dialogue import (
    demo_cases,
    evaluate_dialogues,
    filter_consensus,
    grade_dialogue,
    run_dialogue,
    scripted_responder,
)


def test_filter_consensus_drops_low_endorsement():
    r = Rubric(rubric_id="r", items=[
        RubricItem(item_id="a", text="x", axis="accuracy", consensus=3),
        RubricItem(item_id="b", text="y", axis="communication", consensus=1),
        RubricItem(item_id="c", text="z", axis="safety"),   # unset → kept
    ])
    kept = {it.item_id for it in filter_consensus(r, 2).items}
    assert kept == {"a", "c"}


def test_run_dialogue_collects_transcript():
    case = demo_cases()[0]   # 2 user turns
    history = run_dialogue(case, scripted_responder(case.demo_responses))
    assert [r for r, _ in history] == ["user", "assistant", "user", "assistant"]
    assert history[1][1] == case.demo_responses[0]


def test_good_dialogue_scores_high():
    case = demo_cases()[0]
    score = grade_dialogue(case, scripted_responder(case.demo_responses))
    assert score.overall == 1.0                       # hits every consensus axis, no contraindication
    assert score.per_axis["safety"] == 1.0
    assert "fluff" not in score.per_item              # below-consensus item filtered out


def test_terse_dialogue_scores_low():
    case = demo_cases()[1]
    score = grade_dialogue(case, scripted_responder(case.demo_responses))
    assert score.overall < 0.3
    assert score.per_axis["accuracy"] == 0.0


def test_evaluate_dialogues_reports_hard_subset():
    metrics, per_case = evaluate_dialogues(demo_cases(), model="mock")
    assert metrics["n"] == 2
    assert set(per_case) == {"dlg-ganyu", "dlg-terse"}
    # the hard case is the terse one → hard mean is its (low) score
    assert metrics["hard_mean_overall"] is not None
    assert metrics["hard_mean_overall"] < metrics["mean_overall"]
    assert "safety" in metrics["per_axis"]
