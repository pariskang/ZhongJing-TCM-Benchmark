"""Confidence calibration — ECE / Brier / reliability bins (framework §13).

"In-the-loop safety" needs a model whose stated confidence matches its accuracy.
This elicits an answer **and** a confidence per item, then reports the Expected
Calibration Error (binned |confidence − accuracy|), the Brier score and the
reliability-diagram bins.  Offline the mock answers the first option at a fixed
0.9 confidence — i.e. **over-confident** — so on a half-wrong demo set the ECE is
≈0.4, the failure this metric is meant to expose.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from config import Config, load_config
from llm_client import call_json
from schemas import Category, Question
from utils import ensure_parent, get_logger, resolve_path

_log = get_logger("calibration")


# --------------------------------------------------------------------------- #
# Pure metrics                                                                  #
# --------------------------------------------------------------------------- #


def reliability_bins(confidences: list[float], correct: list[int], n_bins: int = 10) -> list[dict]:
    """Per-bin (mean confidence, accuracy, count) over equal-width confidence bins."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (c > lo or (b == 0 and c >= lo)) and c <= hi]
        if not idx:
            continue
        mc = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(correct[i] for i in idx) / len(idx)
        bins.append({"lo": round(lo, 3), "hi": round(hi, 3), "mean_confidence": round(mc, 4),
                     "accuracy": round(acc, 4), "count": len(idx)})
    return bins


def ece(confidences: list[float], correct: list[int], n_bins: int = 10) -> float:
    """Expected Calibration Error = Σ (n_b/N) · |acc_b − conf_b|."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in reliability_bins(confidences, correct, n_bins):
        total += (b["count"] / n) * abs(b["accuracy"] - b["mean_confidence"])
    return round(total, 4)


def brier_score(confidences: list[float], correct: list[int]) -> float:
    """Mean squared error of confidence as P(correct)."""
    if not confidences:
        return 0.0
    return round(sum((c - y) ** 2 for c, y in zip(confidences, correct)) / len(confidences), 4)


def score_calibration(confidences: list[float], correct: list[int], n_bins: int = 10) -> dict:
    n = len(confidences)
    return {
        "n": n,
        "accuracy": round(sum(correct) / n, 4) if n else 0.0,
        "mean_confidence": round(sum(confidences) / n, 4) if n else 0.0,
        "ece": ece(confidences, correct, n_bins),
        "brier": brier_score(confidences, correct),
        "bins": reliability_bins(confidences, correct, n_bins),
    }


# --------------------------------------------------------------------------- #
# Confidence elicitation                                                        #
# --------------------------------------------------------------------------- #


def elicit(question: Question, model: str, cfg: Optional[Config] = None) -> dict:
    """Ask *model* for an answer + confidence; return ``{pred, confidence, refused, correct}``."""
    cfg = cfg or load_config()
    try:
        tmpl = resolve_path(cfg.get("prompts.confidence_eval")).read_text(encoding="utf-8")
    except Exception:
        tmpl = '题目:\n{question}\n输出 {{"answer":["A"],"confidence":0.8}}'
    opts = "\n".join(f"{k}. {v}" for k, v in question.options.items())
    try:
        data = call_json(tmpl.format(question=f"{question.stem}\n{opts}"), model=model)
    except Exception:  # noqa: BLE001
        return {"pred": [], "confidence": 0.0, "refused": True, "correct": False}
    ans = data.get("answer") or []
    if isinstance(ans, str):
        ans = [ans]
    pred = sorted(str(a) for a in ans)
    try:
        conf = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    except Exception:  # noqa: BLE001
        conf = 0.5
    correct = bool(pred) and pred == sorted(question.answer)
    return {"pred": pred, "confidence": conf, "refused": not pred, "correct": correct}


def evaluate_calibration(questions: list[Question], model: str = "mock",
                         cfg: Optional[Config] = None, n_bins: int = 10) -> dict:
    """Elicit confidence over *questions* and score calibration (refusals excluded)."""
    confs, correct = [], []
    for q in questions:
        out = elicit(q, model, cfg)
        if out["refused"]:
            continue
        confs.append(out["confidence"])
        correct.append(int(out["correct"]))
    return score_calibration(confs, correct, n_bins)


# --------------------------------------------------------------------------- #
# Demo + orchestration                                                          #
# --------------------------------------------------------------------------- #

_OPTS = {"A": "肝郁气滞证", "B": "里热实证", "C": "脾胃虚寒证", "D": "气血两虚证"}


def _q(stem: str, gold: str) -> Question:
    return Question(question_id=str(uuid.uuid4()), source_passage_id="cal",
                    category=Category.COMMON_DISEASE, topic_id=0, type="single_choice",
                    difficulty="advanced", stem=stem, options=dict(_OPTS), answer=[gold],
                    explanation="")


def demo_questions() -> list[Question]:
    # Two the mock gets right (gold A) and two it gets wrong (gold B/C) — all at 0.9.
    return [
        _q("胸胁胀痛、善太息、脉弦。证属？", "A"),
        _q("情志抑郁、乳房胀痛、脉弦。证属？", "A"),
        _q("壮热、口渴、舌红苔黄、脉数。证属？", "B"),
        _q("脘腹冷痛、喜温喜按、脉沉迟。证属？", "C"),
    ]


def run(model: str = "mock", cfg: Optional[Config] = None) -> dict:
    """Run confidence calibration for *model* over the demo set → ``results/``."""
    cfg = cfg or load_config()
    metrics = evaluate_calibration(demo_questions(), model, cfg)
    out = ensure_parent(cfg.path("paths.results_dir") / f"calibration_{_slug(model)}.json")
    out.write_text(json.dumps({"model": model, "metrics": metrics}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    _log.info("calibration[%s] ece=%s brier=%s acc=%s conf=%s", model, metrics["ece"],
              metrics["brier"], metrics["accuracy"], metrics["mean_confidence"])
    return metrics


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


if __name__ == "__main__":
    run()
