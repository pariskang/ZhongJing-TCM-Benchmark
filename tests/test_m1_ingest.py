"""Tests for M1 ingestion & cleaning."""
from pathlib import Path

from m1_ingest import clean, dedup, ingest, parse_filename


def test_clean_strips_boilerplate():
    raw = (
        "点击上方蓝字关注我们\n\n"
        "小柴胡汤出自《伤寒论》，主治少阳证。\n\n"
        "长按识别二维码关注\n阅读 10万+\n点赞 分享 在看\n"
        "免责声明：仅供学习。\nhttps://example.com/x [图片]"
    )
    out = clean(raw)
    assert "小柴胡汤" in out
    for noise in ("点击上方", "二维码", "阅读 10万", "https://", "[图片]", "免责声明"):
        assert noise not in out


def test_clean_collapses_punctuation_and_space():
    assert clean("好。。。   多空格") == "好。 多空格"


def test_parse_filename():
    assert parse_filename(Path("医承有道__小柴胡汤.txt")) == ("医承有道", "小柴胡汤")
    assert parse_filename(Path("no_account.txt")) == (None, "no_account")


def test_ingest_and_dedup(tmp_path):
    body = "小柴胡汤出自伤寒论，和解少阳，主治往来寒热、胸胁苦满、口苦咽干，脉弦。" * 5
    (tmp_path / "示例__a.txt").write_text("点击上方关注\n" + body, encoding="utf-8")
    (tmp_path / "示例__b.txt").write_text(body, encoding="utf-8")  # near-duplicate
    (tmp_path / "示例__short.txt").write_text("太短", encoding="utf-8")  # dropped

    arts = ingest(tmp_path, min_chars=50)
    assert len(arts) == 2  # short one dropped
    assert all(a.char_count >= 50 for a in arts)

    kept = dedup(arts, threshold=0.8)
    assert len(kept) == 1  # near-duplicate removed
