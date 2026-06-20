"""Tests for M2 quality scoring & gating."""
import uuid

from m2_quality import heuristic_gate, score_article, tcm_density
from schemas import Article


def _article(text: str) -> Article:
    return Article(
        article_id=str(uuid.uuid4()), source_file="t.txt",
        raw_text=text, clean_text=text, char_count=len(text),
    )


def test_tcm_density_high_vs_low():
    tcm = "阴阳五行气血经络脏腑，辨证论治，小柴胡汤和解少阳，舌红苔黄脉弦。" * 3
    plain = "今天天气很好，我们一起去公园散步，然后吃了午饭，下午看了电影。" * 3
    assert tcm_density(tcm) > tcm_density(plain)
    assert tcm_density(tcm) >= 0.04


def test_heuristic_gate_rejects_short_and_nontcm():
    short = _article("阴阳五行")  # too short
    assert heuristic_gate(short) is False

    plain = _article("今天天气很好，我们一起去公园散步玩耍聊天吃饭看电影逛街购物。" * 10)
    assert heuristic_gate(plain) is False  # low TCM density


def test_heuristic_gate_accepts_tcm():
    tcm = _article(
        "小柴胡汤出自伤寒论，和解少阳，主治往来寒热、胸胁苦满、口苦咽干、脉弦，"
        "辨证论治当抓少阳枢机不利之象，方中柴胡黄芩相配，一散一清。" * 6
    )
    assert heuristic_gate(tcm) is True
    assert tcm.tcm_density is not None


def test_score_article_mock():
    art = _article("阴阳五行气血，小柴胡汤和解少阳，辨证论治。" * 10)
    score = score_article(art, model="mock")
    assert 0 <= score.overall <= 10
    assert abs(score.overall - (score.professionalism + score.popularization + score.practicality) / 3) < 1e-6
