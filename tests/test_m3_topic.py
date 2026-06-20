"""Tests for M3 chunking & tokenisation (BERTopic stack not exercised here)."""
import uuid

from m3_topic import chunk_article, jieba_tok
from schemas import Article


def _article(text: str) -> Article:
    return Article(
        article_id=str(uuid.uuid4()), source_file="t.txt",
        raw_text=text, clean_text=text, char_count=len(text),
    )


def test_chunk_paragraph_split():
    art = _article("第一段，讲述阴阳五行的基本理论内容足够长以通过最小长度限制要求。\n\n"
                   "第二段，讲述脏腑经络气血津液的相互关系也足够长以通过过滤阈值。")
    passages = chunk_article(art, max_len=400, min_len=10)
    assert len(passages) == 2
    assert all(p.article_id == art.article_id for p in passages)


def test_chunk_sliding_window_on_long_paragraph():
    long_para = "中医" * 500  # 1000 chars, single paragraph
    art = _article(long_para)
    passages = chunk_article(art, max_len=400, overlap=80, min_len=10)
    assert len(passages) >= 2  # windowed
    assert all(len(p.text) <= 400 for p in passages)


def test_chunk_drops_tiny():
    art = _article("短\n\n" + "阴阳五行气血经络脏腑藏象精气津液正气邪气病机气机" * 3)
    passages = chunk_article(art, min_len=50)
    assert all(len(p.text) > 50 for p in passages)


def test_jieba_tok_removes_stopwords():
    toks = jieba_tok("我们的脾胃虚弱需要健脾益气")
    assert "的" not in toks
    assert all(len(t) > 1 for t in toks)
