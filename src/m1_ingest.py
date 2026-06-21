"""M1 — Data ingestion & cleaning.

Turns messy public-account documents (``.txt`` / ``.html`` / ``.docx``) into
clean, structured :class:`~schemas.Article` records and de-duplicates
near-identical reposts with MinHash/LSH (a key defence against benchmark
leakage, manual §1).

Heavy/optional deps (``opencc``, ``jieba``, ``datasketch``, ``python-docx``,
``beautifulsoup4``) are imported lazily so the module stays importable in
minimal environments; ``.docx``/``.html`` also have dependency-free fallbacks.
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

#: File types M1 knows how to read.
SUPPORTED_EXTENSIONS = (".txt", ".md", ".html", ".htm", ".docx")

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


def _read_html(path: Path) -> str:
    """Extract visible text from an ``.html`` file (bs4, with a regex fallback)."""
    raw = read_text(path)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception as exc:  # pragma: no cover - optional dep / parse fallback
        _log.debug("bs4 unavailable for %s (%s); using regex strip", path.name, exc)
        import html as _html

        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", raw)
        text = re.sub(r"(?s)<[^>]+>", "\n", text)
        return _html.unescape(text)


def _read_docx(path: Path) -> str:
    """Extract paragraph text from a ``.docx`` (python-docx, with a zip fallback)."""
    try:
        import docx  # python-docx

        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    except Exception as exc:  # pragma: no cover - optional dep / parse fallback
        _log.debug("python-docx unavailable for %s (%s); reading XML directly", path.name, exc)
        try:
            import html as _html
            import zipfile

            with zipfile.ZipFile(path) as zf:
                xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
            xml = re.sub(r"</w:p>", "\n", xml)          # paragraph → newline
            xml = re.sub(r"<[^>]+>", "", xml)            # drop all remaining tags
            return _html.unescape(xml)
        except Exception as exc2:  # noqa: BLE001
            _log.warning("could not read docx %s (%s)", path.name, exc2)
            return ""


def read_document(path: Path) -> str:
    """Read ``.txt`` / ``.md`` / ``.html`` / ``.docx`` into raw text by extension."""
    ext = path.suffix.lower()
    if ext in (".html", ".htm"):
        return _read_html(path)
    if ext == ".docx":
        return _read_docx(path)
    return read_text(path)


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


#: Leading ``[账号]`` / ``【账号】`` marker.
_ACCOUNT_RE = re.compile(r"^\s*[\[【]\s*([^\]】]+?)\s*[\]】]")
#: A ``2023-03-10`` / ``2023.3.10`` / ``2023年3月10日`` style date anywhere in the name.
_DATE_RE = re.compile(r"(20\d{2})\s*[.\-_/年]\s*(\d{1,2})\s*[.\-_/月]\s*(\d{1,2})")
#: Leading separators / punctuation / symbols to peel off before the title.
_LEAD_JUNK_RE = re.compile(r"^[\s\-—–·•|:：、，,.。_~～*#＃]+")


def _strip_known_ext(name: str) -> str:
    """Drop a single trailing supported extension (case-insensitive)."""
    low = name.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def parse_filename(path: Path) -> tuple[Optional[str], str, Optional[str]]:
    """Parse ``(account, title, publish_date)`` from a possibly messy filename.

    Two conventions are supported:

    * legacy ``账号__标题`` (double underscore), and
    * ``[账号] - 2023-03-10 标题`` with arbitrary leading brackets / dates /
      symbols — the **title is taken from the trailing part** of the name, after
      peeling the leading account marker, date and separators.

    Example::

        "[中医书友会] - 2023-03-10 有多少大夫，正拿着“中医”的金饭碗讨饭.第22期.docx"
        → ("中医书友会", "有多少大夫，正拿着“中医”的金饭碗讨饭.第22期", "2023-03-10")
    """
    stem = _strip_known_ext(path.name)

    # Legacy 账号__标题 convention.
    if "__" in stem:
        account, title = stem.split("__", 1)
        return account.strip() or None, (title.strip() or stem.strip()), None

    account: Optional[str] = None
    m = _ACCOUNT_RE.search(stem)
    if m:
        account = m.group(1).strip() or None
        stem = stem[m.end():]

    date: Optional[str] = None
    dm = _DATE_RE.search(stem)
    if dm:
        y, mo, d = dm.group(1), int(dm.group(2)), int(dm.group(3))
        date = f"{y}-{mo:02d}-{d:02d}"
        stem = stem[: dm.start()] + " " + stem[dm.end():]   # remove date from title

    title = _LEAD_JUNK_RE.sub("", stem)
    title = re.sub(r"\s+", " ", title).strip()
    return account, (title or path.stem), date


def ingest(
    raw_dir: "str | Path" = "data/raw",
    min_chars: int = 100,
    extensions: "tuple[str, ...] | None" = None,
) -> list[Article]:
    """Read every supported document under *raw_dir* into cleaned Articles.

    Recurses into sub-directories and accepts ``.txt`` / ``.md`` / ``.html`` /
    ``.docx`` (configurable via *extensions*).
    """
    exts = tuple(e.lower() for e in (extensions or SUPPORTED_EXTENSIONS))
    arts: list[Article] = []
    base = resolve_path(raw_dir)
    files = sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    for p in files:
        raw = read_document(p)
        clean_text = clean(raw)
        if len(clean_text) < min_chars:        # 过短直接丢
            _log.debug("skip %s (len=%d < %d)", p.name, len(clean_text), min_chars)
            continue
        account, title, date = parse_filename(p)
        arts.append(
            Article(
                article_id=str(uuid.uuid4()),
                source_file=p.name,
                account=account,
                title=title,
                publish_date=date,
                raw_text=raw,
                clean_text=clean_text,
                char_count=len(clean_text),
            )
        )
    _log.info("ingested %d articles from %s (%d candidate files)", len(arts), base, len(files))
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
    exts = cfg.get("ingest.extensions") or SUPPORTED_EXTENSIONS
    arts = ingest(
        cfg.path("paths.raw_dir"),
        min_chars=cfg.get("ingest.min_chars", 100),
        extensions=tuple(exts),
    )
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
