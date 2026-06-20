"""M3 — Passage chunking + BERTopic modelling.

Implements the paper's Eq. 1–3 topic pipeline:

* Eq. 1 — sentence-BERT embeddings (``BAAI/bge-large-zh-v1.5``)
* Eq. 2 — HDBSCAN density clustering (on UMAP-reduced vectors)
* Eq. 3 — c-TF-IDF topic representation (jieba-tokenised ``CountVectorizer``)

The heavy stack (``bertopic``, ``sentence-transformers``, ``umap``, ``hdbscan``)
is imported lazily so ``chunk_article`` and friends stay usable without it.
"""
from __future__ import annotations

import re
import uuid
from functools import lru_cache
from typing import Optional

from config import Config, load_config
from schemas import Article, Passage
from utils import get_logger, load_jsonl_as, read_lines, save_jsonl

_log = get_logger("m3_topic")


# --------------------------------------------------------------------------- #
# Semantic chunking                                                             #
# --------------------------------------------------------------------------- #


def chunk_article(article: Article, max_len: int = 400, overlap: int = 80,
                  min_len: int = 50) -> list[Passage]:
    """Split an article into passages: paragraph-first, sliding-window for long ones."""
    paras = [p.strip() for p in re.split(r"\n{2,}", article.clean_text) if p.strip()]
    texts: list[str] = []
    for para in paras:
        if len(para) <= max_len:
            texts.append(para)
        else:
            step = max(1, max_len - overlap)
            for i in range(0, len(para), step):
                texts.append(para[i : i + max_len])
    return [
        Passage(passage_id=str(uuid.uuid4()), article_id=article.article_id, text=t)
        for t in texts
        if len(t) > min_len
    ]


def chunk_articles(articles: list[Article], cfg: Optional[Config] = None) -> list[Passage]:
    cfg = cfg or load_config()
    passages: list[Passage] = []
    for a in articles:
        passages.extend(
            chunk_article(
                a,
                max_len=cfg.get("topic.max_len", 400),
                overlap=cfg.get("topic.overlap", 80),
                min_len=cfg.get("topic.min_passage_len", 50),
            )
        )
    _log.info("chunked %d articles -> %d passages", len(articles), len(passages))
    return passages


# --------------------------------------------------------------------------- #
# Tokeniser for c-TF-IDF                                                         #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _stopwords(path: str = "lexicons/stopwords_zh.txt") -> frozenset:
    return frozenset(read_lines(path))


def jieba_tok(text: str) -> list[str]:
    """Tokenise for the c-TF-IDF ``CountVectorizer`` (drop stopwords / 1-char)."""
    import jieba

    stop = _stopwords()
    return [w for w in jieba.cut(text) if w.strip() and w not in stop and len(w) > 1]


# --------------------------------------------------------------------------- #
# BERTopic                                                                       #
# --------------------------------------------------------------------------- #


def build_topic_model(cfg: Optional[Config] = None):
    """Construct a Chinese-configured BERTopic model (Eq. 1–3)."""
    cfg = cfg or load_config()
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    rs = cfg.get("topic.random_state", 42)
    emb = SentenceTransformer(cfg.get("topic.embedding_model", "BAAI/bge-large-zh-v1.5"))
    umap = UMAP(
        n_neighbors=cfg.get("topic.umap_neighbors", 15),
        n_components=cfg.get("topic.umap_components", 5),
        min_dist=0.0,
        metric="cosine",
        random_state=rs,
    )
    hdb = HDBSCAN(
        min_cluster_size=cfg.get("topic.min_cluster_size", 15),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vec = CountVectorizer(tokenizer=jieba_tok)
    return BERTopic(
        embedding_model=emb,
        umap_model=umap,
        hdbscan_model=hdb,
        vectorizer_model=vec,
        calculate_probabilities=True,
        verbose=True,
    )


def fit_topics(passages: list[Passage], cfg: Optional[Config] = None):
    """Fit BERTopic and write ``topic_id`` / ``topic_keywords`` back onto passages."""
    cfg = cfg or load_config()
    docs = [p.text for p in passages]
    model = build_topic_model(cfg)
    topics, _probs = model.fit_transform(docs)

    # Optionally converge to the paper's ~175-topic scale.
    nr_topics = cfg.get("topic.nr_topics")
    if nr_topics:
        try:
            model.reduce_topics(docs, nr_topics=int(nr_topics))
            topics = model.topics_
        except Exception as exc:  # noqa: BLE001
            _log.warning("reduce_topics failed (%s); keeping raw topics", exc)

    # Reassign HDBSCAN outliers (-1) to their nearest topic if requested.
    if cfg.get("topic.reduce_outliers", True):
        try:
            topics = model.reduce_outliers(docs, topics)
            model.update_topics(docs, topics=topics, vectorizer_model=CountVectorizer_(jieba_tok))
        except Exception as exc:  # noqa: BLE001
            _log.warning("reduce_outliers failed (%s); keeping outliers", exc)

    for p, t in zip(passages, topics):
        p.topic_id = int(t)
        p.topic_keywords = [w for w, _ in (model.get_topic(t) or [])][:10] if t != -1 else []
    n_topics = len({p.topic_id for p in passages if p.topic_id != -1})
    _log.info("fitted %d topics over %d passages", n_topics, len(passages))
    return model, passages


def CountVectorizer_(tokenizer):  # tiny helper to avoid re-importing inline
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(tokenizer=tokenizer)


def run(cfg: Optional[Config] = None):
    """Chunk quality-passed articles, fit topics, persist passages + model."""
    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    scored = interim / "articles_scored.jsonl"
    src = scored if scored.exists() else (interim / "articles.jsonl")
    articles = load_jsonl_as(src, Article)
    if scored.exists():
        articles = [a for a in articles if a.quality_passed]
    passages = chunk_articles(articles, cfg)

    model, passages = fit_topics(passages, cfg)
    save_jsonl(passages, interim / "passages_topiced.jsonl")
    try:
        model.save(str(cfg.path("paths.topic_model_dir")), serialization="safetensors",
                   save_embedding_model=False)
    except Exception as exc:  # noqa: BLE001
        _log.warning("could not persist topic model: %s", exc)
    return model, passages


if __name__ == "__main__":
    run()
