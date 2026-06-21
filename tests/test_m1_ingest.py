"""Tests for M1 ingestion & cleaning."""
import zipfile
from pathlib import Path

from m1_ingest import clean, dedup, ingest, parse_filename, read_document


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


def test_parse_filename_legacy():
    assert parse_filename(Path("医承有道__小柴胡汤.txt")) == ("医承有道", "小柴胡汤", None)
    assert parse_filename(Path("no_account.txt")) == (None, "no_account", None)


def test_parse_filename_messy_prefix():
    # account in [..], a date, leading " - " symbols; title from the tail; .docx
    name = "[中医书友会] - 2023-03-10 有多少大夫，正拿着“中医”的金饭碗讨饭.第22期.docx"
    account, title, date = parse_filename(Path(name))
    assert account == "中医书友会"
    assert date == "2023-03-10"
    assert title == "有多少大夫，正拿着“中医”的金饭碗讨饭.第22期"


def test_parse_filename_bracket_and_dotted_date():
    account, title, date = parse_filename(Path("【杏林学堂】2022.5.1·谈谈艾灸养生.html"))
    assert account == "杏林学堂"
    assert date == "2022-05-01"
    assert title == "谈谈艾灸养生"


def test_read_document_html(tmp_path):
    p = tmp_path / "a.html"
    p.write_text(
        "<html><head><style>x{}</style></head><body><h1>小柴胡汤</h1>"
        "<p>和解少阳，主治往来寒热。</p><script>ignore()</script></body></html>",
        encoding="utf-8",
    )
    text = read_document(p)
    assert "小柴胡汤" in text and "和解少阳" in text
    assert "ignore" not in text and "x{}" not in text


def test_read_document_docx_fallback(tmp_path):
    # Build a minimal valid .docx (zip) so the XML fallback path works without python-docx.
    p = tmp_path / "b.docx"
    doc_xml = (
        '<?xml version="1.0"?>'
        "<w:document xmlns:w='x'><w:body>"
        "<w:p><w:r><w:t>脾胃虚弱</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>健脾益气，主治食少乏力。</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)
    text = read_document(p)
    assert "脾胃虚弱" in text and "健脾益气" in text


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


def test_ingest_mixed_extensions(tmp_path):
    body = "小柴胡汤出自伤寒论，和解少阳，主治往来寒热、胸胁苦满、口苦咽干，脉弦。" * 3
    (tmp_path / "[账号A] - 2023-03-10 柴胡汤.txt").write_text(body, encoding="utf-8")
    (tmp_path / "[账号B]脾胃调理.html").write_text(
        f"<html><body><p>{body}</p></body></html>", encoding="utf-8"
    )
    (tmp_path / "ignored.pdf").write_text("not read", encoding="utf-8")  # unsupported

    arts = ingest(tmp_path, min_chars=50)
    assert len(arts) == 2  # txt + html; pdf ignored
    by_account = {a.account: a for a in arts}
    assert set(by_account) == {"账号A", "账号B"}
    assert by_account["账号A"].publish_date == "2023-03-10"
