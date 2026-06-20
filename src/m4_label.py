"""M4 — Nine-category mapping + human-in-the-loop review.

Maps each BERTopic topic to one of the nine TCM categories (the paper's labelling
function ``f``, Eq. 4) by anchor-term voting, emits **topic cards** for physician
review, and measures inter-annotator agreement with Cohen's kappa (Eq. 9).

Because mapping is done at the *topic* level (~175 topics, not every passage),
the physician workload is small; once agreed, the label is propagated to all
passages under that topic.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import yaml

from config import Config, load_config
from schemas import Category, Passage
from utils import get_logger, load_jsonl_as, resolve_path, save_jsonl

_log = get_logger("m4_label")


def load_anchors(path: str = "lexicons/category_anchors.yaml") -> dict[str, list[str]]:
    with resolve_path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def suggest_category(topic_keywords: list[str], anchors: dict[str, list[str]]) -> Optional[Category]:
    """Vote a category from topic keywords; zero-hit/tie → ``None`` (human review)."""
    scores = {
        cat: sum(
            1 for kw in topic_keywords if any(a in kw or kw in a for a in anchor_terms)
        )
        for cat, anchor_terms in anchors.items()
    }
    if not scores:
        return None
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return None
    # Tie between two categories → defer to a physician.
    if sum(1 for v in scores.values() if v == best_score) > 1:
        return None
    return Category.from_label(best)


def build_topic_cards(passages: list[Passage], anchors: dict[str, list[str]],
                      reps: int = 3) -> list[dict]:
    """One review card per topic: keywords, representative passages, suggestion."""
    by_topic: dict[int, list[Passage]] = defaultdict(list)
    for p in passages:
        if p.topic_id is not None and p.topic_id != -1:
            by_topic[p.topic_id].append(p)

    cards: list[dict] = []
    for topic_id in sorted(by_topic):
        group = by_topic[topic_id]
        keywords = group[0].topic_keywords or []
        suggestion = suggest_category(keywords, anchors)
        examples = sorted(group, key=lambda p: len(p.text), reverse=True)[:reps]
        cards.append(
            {
                "topic_id": topic_id,
                "size": len(group),
                "keywords": ", ".join(keywords),
                "suggested_category": suggestion.value if suggestion else "",
                "rep_passages": " ||| ".join(p.text[:120] for p in examples),
                # blank columns for the two physician annotators:
                "annotator_a": "",
                "annotator_b": "",
                "final_category": suggestion.value if suggestion else "",
            }
        )
    return cards


def assign_categories(passages: list[Passage], topic_to_cat: dict[int, Category]) -> list[Passage]:
    """Propagate per-topic category labels to every passage."""
    for p in passages:
        cat = topic_to_cat.get(p.topic_id)
        if cat is not None:
            p.category = cat
    return passages


def review_agreement(labels_a: list[str], labels_b: list[str], threshold: float = 0.8) -> float:
    """Eq. 9 — Cohen's kappa; asserts ``κ ≥ threshold`` (good agreement)."""
    from sklearn.metrics import cohen_kappa_score

    k = cohen_kappa_score(labels_a, labels_b)
    assert k >= threshold, f"标注一致性不足 κ={k:.3f}，需统一标注规范后重标"
    return k


def _load_human_labels(cards_path) -> dict[int, Category]:
    """Read back a reviewed ``topic_cards.csv`` if present (uses ``final_category``)."""
    import csv

    mapping: dict[int, Category] = {}
    p = resolve_path(cards_path)
    if not p.exists():
        return mapping
    with p.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cat = Category.from_label(row.get("final_category", "").strip())
            if cat is not None:
                mapping[int(row["topic_id"])] = cat
    return mapping


def run(cfg: Optional[Config] = None) -> list[Passage]:
    """Auto-suggest categories, emit topic cards, propagate labels to passages."""
    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    passages = load_jsonl_as(interim / "passages_topiced.jsonl", Passage)
    anchors = load_anchors(cfg.get("lexicons.category_anchors"))

    cards = build_topic_cards(passages, anchors)
    _write_cards_csv(cards, interim / "topic_cards.csv")

    # Prefer human-reviewed labels if a completed card file exists; else auto.
    topic_to_cat = _load_human_labels(interim / "topic_cards_reviewed.csv")
    if not topic_to_cat:
        topic_to_cat = {
            c["topic_id"]: Category.from_label(c["final_category"])
            for c in cards
            if c["final_category"]
        }
        topic_to_cat = {k: v for k, v in topic_to_cat.items() if v is not None}

    passages = assign_categories(passages, topic_to_cat)
    labelled = [p for p in passages if p.category is not None]
    _log.info(
        "labelled %d/%d passages across %d topics",
        len(labelled), len(passages), len(topic_to_cat),
    )
    save_jsonl(passages, interim / "passages_labeled.jsonl")
    return passages


def _write_cards_csv(cards: list[dict], path) -> None:
    import csv

    if not cards:
        return
    from utils import ensure_parent

    out = ensure_parent(path)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cards[0].keys()))
        writer.writeheader()
        writer.writerows(cards)
    _log.info("wrote %d topic cards -> %s", len(cards), out)


if __name__ == "__main__":
    run()
