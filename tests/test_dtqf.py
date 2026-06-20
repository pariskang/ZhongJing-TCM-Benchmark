"""Tests for M6 — DTQF dynamic filtering (core algorithm)."""
from m6_dtqf import KeywordExtractor, dtqf_filter, review_question


def test_review_rejects_incomplete_clinical(make_q, kw_ext):
    bad = make_q(category="常见病辨证论治", stem="患者来诊，请辨证论治给出诊断结论。")
    good = make_q(category="常见病辨证论治",
                  stem="患者发热恶寒、舌红苔黄、脉数、口苦咽干，请辨证论治。")
    assert review_question(bad, kw_ext) is False     # missing symptoms
    assert review_question(good, kw_ext) is True


def test_review_format_checks(make_q):
    # single-choice with <4 options is rejected
    assert review_question(make_q(options={"A": "甲", "B": "乙"})) is False
    # empty answer rejected
    assert review_question(make_q(answer=[])) is False
    # short answer without reference answer rejected
    assert review_question(make_q(type="short_answer", reference_answer="")) is False
    # answer referencing a non-existent option rejected
    assert review_question(make_q(answer=["Z"])) is False


def test_dtqf_removes_incomplete_clinical(make_q, kw_ext):
    bad = make_q(category="常见病辨证论治", stem="患者来诊，请辨证论治给出最终结论。")
    good = make_q(category="常见病辨证论治",
                  stem="患者发热恶寒、舌红苔黄、脉数、口苦咽干，请辨证论治。")
    survivors = dtqf_filter([bad, good], kw_ext, S=10, max_iter=3, seed=0)
    assert good in survivors and bad not in survivors
    assert good.qc_passed is True


def test_dtqf_keeps_valid_questions(make_q, kw_ext):
    qs = [make_q(topic_id=i % 3) for i in range(30)]
    survivors = dtqf_filter(qs, kw_ext, S=10, max_iter=5, seed=1)
    assert len(survivors) == 30  # nothing wrong with them


def test_keyword_extractor_basic():
    corpus = [["柴胡", "黄芩", "少阳", "和解"], ["发热", "恶寒", "舌红", "脉数"]]
    ke = KeywordExtractor(corpus, min_count=1)
    kws = ke.extract("患者发热恶寒，舌红脉数")
    assert isinstance(kws, list)
