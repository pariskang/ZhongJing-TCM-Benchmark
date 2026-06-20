"""M7 — Dataset assembly, packaging & statistics reproduction.

* counts ``stem / answer / explanation`` tokens per question,
* writes the full set and a stratified-10% **diagnostic** split,
* emits the ``dataset_card`` table (paper Table ``data_overview``),
* renders Figure 2 (polar token distribution) and Figure 6 (3-D UMAP).
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Optional

from config import Config, load_config
from schemas import Question
from utils import ensure_parent, get_logger, load_jsonl_as, save_jsonl

_log = get_logger("m7_assemble")

_CJK_RE = re.compile(r"[一-鿿]")


@lru_cache(maxsize=2)
def _encoder(name: str = "cl100k_base"):
    """Return a tiktoken encoder, or ``None`` if unavailable (offline fallback)."""
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as exc:  # pragma: no cover - network/optional
        _log.warning("tiktoken encoding %s unavailable (%s); using heuristic", name, exc)
        return None


def _encode_len(text: str, name: str = "cl100k_base") -> int:
    """Token count via tiktoken, with a deterministic CJK-aware fallback."""
    if not text:
        return 0
    enc = _encoder(name)
    if enc is not None:
        return len(enc.encode(text))
    cjk = len(_CJK_RE.findall(text))
    non_cjk = len([w for w in _CJK_RE.sub(" ", text).split() if w])
    return cjk + non_cjk


def count_tokens(q: Question, encoding: str = "cl100k_base") -> Question:
    """Populate ``q.tokens`` with the three-span token counts."""
    answer_text = " ".join(q.answer) or (q.reference_answer or "")
    q.tokens = {
        "stem": _encode_len(q.stem, encoding),
        "answer": _encode_len(answer_text, encoding),
        "explanation": _encode_len(q.explanation, encoding),
    }
    return q


def stratified_sample(questions: list[Question], frac: float = 0.10,
                      seed: Optional[int] = 42) -> list[Question]:
    """诊断模式:按 (category, difficulty, type) 分层抽样 *frac*。"""
    rng = random.Random(seed)
    buckets: dict[tuple, list[Question]] = defaultdict(list)
    for q in questions:
        buckets[(q.category, q.difficulty, q.type)].append(q)
    out: list[Question] = []
    for grp in buckets.values():
        k = max(1, round(len(grp) * frac))
        out += rng.sample(grp, min(k, len(grp)))
    return out


def dataset_card(questions: list[Question]):
    """Per-category overview: #topics (NoT), #questions (N), total stem tokens."""
    import pandas as pd

    df = pd.DataFrame([q.model_dump(mode="json") for q in questions])
    if df.empty:
        return df
    card = (
        df.groupby("category")
        .agg(
            NoT=("topic_id", "nunique"),
            N=("question_id", "count"),
            Tokens=("tokens", lambda s: int(sum(d.get("stem", 0) for d in s))),
        )
        .sort_values("N", ascending=False)
    )
    return card


def run(cfg: Optional[Config] = None) -> list[Question]:
    """Token-count, split and card the QC'd questions into ``data/final``."""
    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    final = cfg.path("paths.final_dir")
    encoding = cfg.get("assemble.token_encoding", "cl100k_base")

    questions = load_jsonl_as(interim / "questions_qc.jsonl", Question)
    questions = [count_tokens(q, encoding) for q in questions]

    save_jsonl(questions, final / "zhongjing_tcm_full.jsonl")
    diagnostic = stratified_sample(questions, frac=cfg.get("assemble.diagnostic_frac", 0.10))
    save_jsonl(diagnostic, final / "zhongjing_tcm_diagnostic.jsonl")

    card = dataset_card(questions)
    if not card.empty:
        card_path = ensure_parent(final / "dataset_card.csv")
        card.to_csv(card_path, encoding="utf-8")
        _log.info("dataset card:\n%s", card.to_string())

    try:
        plot_token_polar(questions, cfg)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Figure 2 (polar) skipped: %s", exc)
    return questions


# --------------------------------------------------------------------------- #
# Figures                                                                       #
# --------------------------------------------------------------------------- #


def plot_token_polar(questions: list[Question], cfg: Optional[Config] = None):
    """Figure 2 — polar distribution of Q/A/Explanation tokens per question type."""
    import matplotlib

    matplotlib.use("Agg")
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    cfg = cfg or load_config()
    types = ["single_choice", "multiple_response", "short_answer"]
    spans = ["stem", "answer", "explanation"]
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(15, 5))
    for ax, qtype in zip(axes, types):
        qs = [q for q in questions if q.type == qtype]
        ax.set_title(qtype)
        for span in spans:
            vals = sorted(q.tokens.get(span, 0) for q in qs)
            if not vals:
                continue
            theta = np.linspace(0, 2 * math.pi, len(vals), endpoint=False)
            ax.plot(theta, vals, label=span, linewidth=1)
        ax.legend(loc="upper right", fontsize=7)
    out = ensure_parent(cfg.path("paths.results_dir") / "figures" / "figure2_token_polar.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log.info("wrote %s", out)
    return out


def plot_umap_3d(passages, cfg: Optional[Config] = None):
    """Figure 6 — 3-D UMAP projection of passage embeddings, coloured by topic."""
    cfg = cfg or load_config()
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    texts = [p.text for p in passages]
    emb = SentenceTransformer(cfg.get("topic.embedding_model")).encode(texts, show_progress_bar=True)
    coords = UMAP(n_components=3, metric="cosine", random_state=cfg.get("topic.random_state", 42)).fit_transform(emb)
    try:
        import plotly.express as px

        fig = px.scatter_3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            color=[str(p.topic_id) for p in passages],
        )
        out = ensure_parent(cfg.path("paths.results_dir") / "figures" / "figure6_umap3d.html")
        fig.write_html(str(out))
        _log.info("wrote %s", out)
        return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("plotly unavailable (%s); returning coords only", exc)
        return coords


if __name__ == "__main__":
    run()
