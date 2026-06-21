"""M1 — Data ingestion & cleaning.

Turns messy WeChat-public-account ``.txt`` files into clean, structured
:class:`~schemas.Article` records and de-duplicates near-identical reposts with
MinHash/LSH (a key defence against benchmark leakage, manual §1).

Heavy/optional deps (``opencc``, ``jieba``, ``datasketch``) are imported lazily
so the module stays importable in minimal environments.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

from config import Config, load_config
from schemas import Article
from utils import get_logger, resolve_path, save_jsonl

_log = get_logger("m1_ingest")

# --------------------------------------------------------------------------- #
# Boilerplate / noise patterns                                                  #
# --------------------------------------------------------------------------- #

BOILERPLATE = [
    r"点击上方.{0,8}关注",
    r"长按.{0,4}(识别|扫描).{0,6}二维码",
    r"(本文|文章)?(转载|来源|整理)自.{0,30}",
    r"免责声明[:：].*",
    r"版权声明[:：].*",
    r"投稿邮箱[:：]\S+",
    r"商务合作\S*",
    r"阅读\s*\d+",
    r"在看\s*\d+",
    r"点赞|分享|收藏|关注我们",
    r"扫码加.{0,6}(微信|群)",
    r"[●▼◆★☆■]+",
]
BP_RE = re.compile("|".join(BOILERPLATE))
URL_RE = re.compile(r"https?://\S+|www\.\S+")
IMG_RE = re.compile(r"\[图片\]|\[图片占位\]|【图\d*】")
# Most emoji / pictographic ranges (kept out of the cleaned text).
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF️]"
)

_CC = None  # cached OpenCC converter


def _converter():
    """Lazily build a traditional→simplified OpenCC converter (optional dep)."""
    global _CC
    if _CC is None:
        try:
            from opencc import OpenCC

            _CC = OpenCC("t2s")
        except Exception as exc:  # pragma: no cover - optional dep
            _log.warning("opencc unavailable (%s); skipping zh-Hant→zh-Hans", exc)
            _CC = False
    return _CC


def read_text(path: Path) -> str:
    """Read a text file, auto-detecting its encoding."""
    raw = path.read_bytes()
    try:
        import chardet

        enc = chardet.detect(raw)["encoding"] or "utf-8"
    except Exception:  # pragma: no cover - optional dep
        enc = "utf-8"
    return raw.decode(enc, errors="ignore")


def clean(text: str) -> str:
    """Normalise encoding, strip boilerplate/URLs/images, squeeze whitespace."""
    cc = _converter()
    if cc:
        text = cc.convert(text)            # 繁简归一
    text = URL_RE.sub("", text)
    text = IMG_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = BP_RE.sub("", text)
    # 全角空格、连续空白、重复标点归一
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([。！？])\1+", r"\1", text)
    return text.strip()


def parse_filename(path: Path) -> tuple[Optional[str], str]:
    """Filename convention: ``账号__标题.txt`` → (account, title)."""
    stem = path.stem
    if "__" in stem:
        account, title = stem.split("__", 1)
        return account.strip() or None, title.strip()
    return None, stem


def ingest(raw_dir: "str | Path" = "data/raw", min_chars: int = 100) -> list[Article]:
    """Read every ``*.txt`` under *raw_dir* into cleaned :class:`Article` records."""
    arts: list[Article] = []
    base = resolve_path(raw_dir)
    for p in sorted(base.glob("*.txt")):
        raw = read_text(p)
        clean_text = clean(raw)
        if len(clean_text) < min_chars:        # 过短直接丢
            _log.debug("skip %s (len=%d < %d)", p.name, len(clean_text), min_chars)
            continue
        account, title = parse_filename(p)
        arts.append(
            Article(
                article_id=str(uuid.uuid4()),
                source_file=p.name,
                account=account,
                title=title,
                raw_text=raw,
                clean_text=clean_text,
                char_count=len(clean_text),
            )
        )
    _log.info("ingested %d articles from %s", len(arts), base)
    return arts


def dedup(articles: list[Article], threshold: float = 0.85, num_perm: int = 128) -> list[Article]:
    """Remove near-duplicate reposts with MinHash/LSH (Jaccard ≥ *threshold*)."""
    try:
        from datasketch import MinHash, MinHashLSH
    except Exception as exc:  # pragma: no cover - optional dep
        _log.warning("datasketch unavailable (%s); skipping de-duplication", exc)
        return articles
    import jieba

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Article] = []
    for a in articles:
        m = MinHash(num_perm=num_perm)
        for w in jieba.cut(a.clean_text):
            m.update(w.encode("utf-8"))
        if not lsh.query(m):
            lsh.insert(a.article_id, m)
            kept.append(a)
    _log.info("de-dup: %d -> %d articles (threshold=%.2f)", len(articles), len(kept), threshold)
    return kept


def run(cfg: Optional[Config] = None) -> list[Article]:
    """CLI entry point: ingest → de-dup → write ``interim/articles.jsonl``."""
    cfg = cfg or load_config()
    arts = ingest(cfg.path("paths.raw_dir"), min_chars=cfg.get("ingest.min_chars", 100))
    arts = dedup(
        arts,
        threshold=cfg.get("ingest.dedup_threshold", 0.85),
        num_perm=cfg.get("ingest.minhash_perm", 128),
    )
    out = cfg.path("paths.interim_dir") / "articles.jsonl"
    save_jsonl(arts, out)
    return arts


if __name__ == "__main__":
    run()
