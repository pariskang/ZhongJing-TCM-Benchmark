"""Tests for T5 MDT: aggregation, group-vs-individual, amplification, safety."""
from schemas import Category
from t5_mdt import (
    MDTCase,
    aggregate,
    chair_adjudicator,
    demo_cases,
    llm_panel,
    run_mdt_case,
    scripted_agent,
    weighted_majority,
)


def _case(answer="A", red_flag=None):
    return MDTCase(
        case_id="c", category=Category.COMMON_DISEASE,
        stem="…证属？", options={"A": "甲证", "B": "乙证", "C": "丙证", "D": "丁证"},
        answer=[answer], red_flag=red_flag,
    )


# -- aggregation -------------------------------------------------------------- #


def test_weighted_majority_and_chair():
    ops = [
        scripted_agent("辨证", "A", 0.6)(_case()),
        scripted_agent("方剂", "A", 0.6)(_case()),
        scripted_agent("针灸", "B", 0.95)(_case()),
    ]
    assert weighted_majority(ops) == ["A"]        # 1.2 vs 0.95
    assert chair_adjudicator(ops) == ["B"]        # highest-confidence specialist


# -- group vs individual ------------------------------------------------------ #


def test_group_corrects_individual_error():
    case = _case(answer="A")
    agents = [scripted_agent("辨证", "A"), scripted_agent("方剂", "A"), scripted_agent("针灸", "B")]
    res = run_mdt_case(case, agents)
    assert res.group_vote == ["A"] and res.group_correct is True
    assert res.disagreement is True
    assert res.corrected is True                  # one agent (针灸) was wrong; group right
    assert res.amplified is False


def test_group_amplifies_shared_blind_spot():
    case = _case(answer="A")
    agents = [scripted_agent(s, "B") for s in ("辨证", "方剂", "针灸")]  # all wrong, agree
    res = run_mdt_case(case, agents)
    assert res.group_correct is False
    assert res.amplified is True                  # majority shares the wrong answer
    assert res.disagreement is False


def test_group_gain_positive_when_panel_diverse():
    # 3 agents, votes A/A/B, gold A → individuals 2/3, group (A) 1.0 → gain > 0.
    case = _case(answer="A")
    agents = [scripted_agent("辨证", "A"), scripted_agent("方剂", "A"), scripted_agent("针灸", "B")]
    m = aggregate([run_mdt_case(case, agents)])
    assert m["mdt_accuracy"] == 1.0
    assert m["mean_individual_accuracy"] == round(2 / 3, 4)
    assert m["group_gain"] > 0


def test_red_flag_recall():
    case = _case(answer="A", red_flag="真心痛")
    # one specialist explicitly raises the red flag in its rationale
    agents = [
        scripted_agent("辨证", "A", rationale="考虑真心痛，需急诊"),
        scripted_agent("方剂", "A"),
    ]
    m = aggregate([run_mdt_case(case, agents)])
    assert m["red_flag_recall"] == 1.0
    # …and a panel that misses it
    miss = aggregate([run_mdt_case(case, [scripted_agent("辨证", "A"), scripted_agent("方剂", "A")])])
    assert miss["red_flag_recall"] == 0.0


# -- mock panel (homogeneous → amplification) -------------------------------- #


def test_mock_panel_is_homogeneous_and_amplifies():
    # Mock votes the first option label for every specialty → no diversity.
    agents = llm_panel("mock")
    results = [run_mdt_case(c, agents) for c in demo_cases()]
    m = aggregate(results)
    assert m["group_gain"] == 0.0                 # group never beats the individual
    assert m["disagreement_rate"] == 0.0          # homogeneous panel
    # the 里热证 case (gold B) is voted A by all → amplified; red flag missed
    assert m["amplified_rate"] > 0.0
    assert m["red_flag_recall"] == 0.0
