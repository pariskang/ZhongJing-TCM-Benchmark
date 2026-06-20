"""M2 — Quality scoring & gating.

Two-stage, article-level quality control reproducing the paper's expert
three-dimension assessment (Figure 3 radar: Professionalism / Popularization /
Practicality):

1. A cheap **heuristic gate** (length, TCM-term density, promo-spam ratio).
2. An **LLM-as-judge** that scores the three dimensions 0–10.

Articles must clear ``heuristic_gate`` *and* reach the overall threshold to pass.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config import Config, load_config
from llm_client import call_json
from schemas import Article, QualityScore
from utils import get_logger, load_jsonl_as, read_lines, resolve_path, save_jsonl

_log = get_logger("m2_quality")

AD_TERMS = ["报名", "课程", "优惠", "限时", "扫码购买", "原价", "立减", "秒杀"]


@lru_cache(maxsize=4)
def load_tcm_lexicon(path: str = "lexicons/tcm_terms.txt") -> frozenset:
    """Load the TCM-term set (cached)."""
    return frozenset(read_lines(path))


@lru_cache(maxsize=1)
def _ensure_userdict(path: str = "lexicons/tcm_terms.txt") -> bool:
    """Register TCM terms with jieba so they segment as single tokens."""
    try:
        import jieba

        for term in read_lines(path):
            jieba.add_word(term)
        return True
    except Exception as exc:  # pragma: no cover - optional dep
        _log.debug("jieba userdict not loaded (%s)", exc)
        return False


def tcm_density(text: str, lexicon: Optional[frozenset] = None) -> float:
    """TCM-term density = (#tokens that are TCM terms) / (#multi-char tokens)."""
    import jieba

    _ensure_userdict()
    vocab = lexicon if lexicon is not None else load_tcm_lexicon()
    words = [w for w in jieba.cut(text) if len(w) > 1]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in vocab)
    return hits / len(words)


def heuristic_gate(article: Article, cfg: Optional[Config] = None) -> bool:
    """Fast structural filter; also records ``tcm_density`` on the article."""
    cfg = cfg or load_config()
    min_chars = cfg.get("quality.min_chars", 300)
    min_density = cfg.get("quality.min_tcm_density", 0.04)
    max_ad = cfg.get("quality.max_ad_hits", 8)

    if article.char_count < min_chars:
        return False
    density = tcm_density(article.clean_text)
    article.tcm_density = round(density, 4)
    if density < min_density:                       # 非中医内容
        return False
    ad_hits = sum(article.clean_text.count(t) for t in AD_TERMS)
    if ad_hits > max_ad:                            # 软文/纯广告
        return False
    return True


def score_article(article: Article, model: str = "gpt-4o") -> QualityScore:
    """LLM-as-judge three-dimension scoring (0–10 each)."""
    cfg = load_config()
    tmpl = resolve_path(cfg.get("prompts.judge_quality")).read_text(encoding="utf-8")
    prompt = tmpl.format(article_text=article.clean_text[:4000])
    data = call_json(prompt, model=model)
    score = QualityScore(
        professionalism=float(data["professionalism"]),
        popularization=float(data["popularization"]),
        practicality=float(data["practicality"]),
        reason=data.get("reason"),
    ).recompute_overall()
    return score


def run(cfg: Optional[Config] = None, llm_judge: bool = True) -> list[Article]:
    """Score & gate ``interim/articles.jsonl`` → ``interim/articles_scored.jsonl``."""
    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    articles = load_jsonl_as(interim / "articles.jsonl", Article)
    model = cfg.get("quality.judge_model", "gpt-4o")
    threshold = cfg.get("quality.overall_threshold", 6.0)

    for a in articles:
        a.heuristic_passed = heuristic_gate(a, cfg)
        if a.heuristic_passed and llm_judge:
            try:
                a.quality = score_article(a, model=model)
            except Exception as exc:  # noqa: BLE001
                _log.warning("scoring failed for %s: %s", a.article_id, exc)
        overall = a.quality.overall if a.quality else 0.0
        a.quality_passed = bool(a.heuristic_passed and (not llm_judge or overall >= threshold))

    kept = sum(a.quality_passed for a in articles)
    _log.info("quality gate: %d/%d articles passed", kept, len(articles))
    save_jsonl(articles, interim / "articles_scored.jsonl")
    return articles


def account_radar(articles: list[Article]):
    """Aggregate three-dimension means per account (reproduces Figure 3)."""
    import pandas as pd

    rows = [
        {
            "account": a.account or "unknown",
            "professionalism": a.quality.professionalism,
            "popularization": a.quality.popularization,
            "practicality": a.quality.practicality,
        }
        for a in articles
        if a.quality is not None
    ]
    if not rows:
        return pd.DataFrame(columns=["professionalism", "popularization", "practicality"])
    return pd.DataFrame(rows).groupby("account").mean(numeric_only=True)


if __name__ == "__main__":
    run()
