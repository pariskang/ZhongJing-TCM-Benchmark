"""Tests for T4 longitudinal episodes: outcome-dependent evolution + scoring."""
import uuid

from schemas import Category
from t4_longitudinal import (
    EpisodeState,
    LongitudinalEpisode,
    aggregate,
    demo_episodes,
    mcq_expert,
    run_episode,
    scripted_visit_expert,
)


def _smooth():
    return next(e for e in demo_episodes() if e.episode_id == "ep-windcold-smooth")


def _misled():
    return next(e for e in demo_episodes() if e.episode_id == "ep-windcold-misled")


# -- trajectory dynamics ------------------------------------------------------ #


def test_correct_first_visit_resolves_cleanly():
    ep = _smooth()  # s0 answer "A"
    res = run_episode(ep, scripted_visit_expert(["A"]))
    assert len(res.visits) == 1
    assert res.resolved is True and res.failed is False
    assert res.adverse_transitions == 0
    assert res.clean_resolution is True
    assert res.node_accuracy == 1.0


def test_wrong_then_correct_recovers_but_not_clean():
    # s0 wrong ("B") → 入里化热 s1; s1 correct ("A") → resolved.
    ep = _smooth()
    res = run_episode(ep, scripted_visit_expert(["B", "A"]))
    assert len(res.visits) == 2
    assert res.resolved is True
    assert res.adverse_transitions == 1          # worsened en route
    assert res.clean_resolution is False         # cured but harmed the patient first
    assert res.node_accuracy == 0.5
    # syndrome changed s0→s1 and the treatment content changed → adjusted
    assert res.syndrome_changes == 1 and res.adjusted == 1
    assert res.adjustment_recall == 1.0


def test_persistently_wrong_ends_in_failure():
    ep = _smooth()  # s0(wrong)->s1(wrong)->s2(wrong terminal)
    res = run_episode(ep, scripted_visit_expert(["B", "B", "B"]))
    assert res.failed is True and res.resolved is False
    assert res.adverse_transitions == 2          # s0→s1, s1→s2
    assert res.clean_resolution is False


def test_no_adjustment_when_treatment_repeated():
    # An episode where the same treatment content sits under the chosen key in both
    # states; repeating it across a syndrome change → not adjusted.
    cat = Category.COMMON_DISEASE
    ep = LongitudinalEpisode(
        episode_id="ep-stuck", category=cat, start="a", max_visits=3,
        states={
            "a": EpisodeState(state_id="a", syndrome="证甲", presentation="初诊。治法？",
                              options={"A": "辛温解表", "B": "清热泻火"}, answer=["B"],
                              on_correct=None, on_wrong="b"),
            "b": EpisodeState(state_id="b", syndrome="证乙", presentation="复诊。治法？",
                              options={"A": "辛温解表", "B": "清营凉血"}, answer=["B"],
                              on_correct=None, on_wrong=None),
        },
    )
    # picks "A" both visits → content "辛温解表" unchanged despite syndrome change
    res = run_episode(ep, scripted_visit_expert(["A", "A"]))
    assert res.syndrome_changes == 1
    assert res.adjusted == 0
    assert res.adjustment_recall == 0.0


# -- mock end-to-end + aggregate --------------------------------------------- #


def test_mock_paths_through_both_episodes():
    # Mock always answers "A": smooth resolves at visit 1 (clean);
    # misled gets visit1 wrong (A≠B) → 入里化热, visit2 correct (A) → resolved.
    results = [run_episode(ep, mcq_expert("mock")) for ep in demo_episodes()]
    smooth, misled = results
    assert smooth.resolved and smooth.clean_resolution and smooth.node_accuracy == 1.0
    assert misled.resolved and not misled.clean_resolution
    assert misled.adverse_transitions == 1
    assert misled.adjusted == 1                   # changed 辛凉解表 → 清热泻火


def test_aggregate_metrics():
    results = [run_episode(ep, mcq_expert("mock")) for ep in demo_episodes()]
    m = aggregate(results)
    assert m["n"] == 2
    assert m["resolution_rate"] == 1.0
    assert m["clean_resolution_rate"] == 0.5      # only the smooth path is clean
    assert m["node_accuracy"] == 0.75             # (1.0 + 0.5) / 2
    assert m["adjustment_recall"] == 1.0
