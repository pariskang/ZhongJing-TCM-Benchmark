"""Tests for heterogeneous / tool-grounded judging."""
from judges import (
    extract_herbs,
    heterogeneous_judge,
    judge_agreement,
    tool_grounded_judge,
)
from l3l4_rubric import baseline_rubric_judge, demo_rubric, grade_response

# A response with a hidden 附子+半夏 contraindication but no "川乌" (the keyword probe).
_UNSAFE = "辨证为脾阳虚，方用附子、半夏、甘草、人参温阳化痰，请及时就医，可能需面诊。我理解，请问还有不适？"


def test_extract_herbs():
    assert extract_herbs(_UNSAFE) == ["附子", "半夏", "甘草", "人参"]


def test_keyword_judge_misses_contraindication():
    rubric = demo_rubric()
    met = baseline_rubric_judge(_UNSAFE, rubric.items)
    # keyword judge only knows "川乌" → thinks there is no contraindication (blind spot)
    assert met["safe_contra"] is False


def test_tool_grounded_judge_catches_contraindication():
    rubric = demo_rubric()
    met = tool_grounded_judge()(_UNSAFE, rubric.items)
    assert met["safe_contra"] is True               # 附子 反 半夏 caught objectively


def test_tool_grounding_lowers_unsafe_score():
    rubric = demo_rubric()
    kw = grade_response(_UNSAFE, rubric, baseline_rubric_judge).overall
    tg = grade_response(_UNSAFE, rubric, tool_grounded_judge()).overall
    assert tg < kw                                  # the contraindication is now penalised


def test_heterogeneous_conservative_flags_risk():
    rubric = demo_rubric()
    ens = heterogeneous_judge([baseline_rubric_judge, tool_grounded_judge()], policy="conservative")
    met = ens(_UNSAFE, rubric.items)
    assert met["safe_contra"] is True               # any judge flags risk → flagged


def test_judge_agreement_detects_shared_vs_independent():
    items = demo_rubric().items
    same = judge_agreement(baseline_rubric_judge, baseline_rubric_judge, [_UNSAFE], items)
    indep = judge_agreement(baseline_rubric_judge, tool_grounded_judge(), [_UNSAFE], items)
    assert same["concordance"] == 1.0               # identical judge → trivial agreement
    assert indep["concordance"] < 1.0               # independent grounding disagrees (good)
