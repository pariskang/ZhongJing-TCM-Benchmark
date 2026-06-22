"""Abstention probes — A@D (Abstention@Detection), reusing M8 refusal detection.

The framework (§13) asks a deployable model to **abstain precisely when, and only
when, the information is insufficient** — detect the gap *and act on it* (refuse /
ask) rather than hallucinate, without over-abstaining on answerable items.

A probe set mixes **answerable** items (complete info → should answer) and
**unanswerable / missing-premise** items (the decisive 四诊 feature removed →
ambiguous → should abstain).  Each item is run through the ordinary M8
:class:`~m8_evaluate.ModelEvaluator`; its ``refused`` flag (explicit "信息不足…"
via ``REFUSAL_PAT`` *or* no option selected) is the abstention signal.

Metrics: abstention precision / recall (= A@D) / F1 on the unanswerable items,
the **over-abstention rate** on answerable items, and answerable accuracy.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from pydantic import BaseModel

from config import Config, load_config
from schemas import Category, Question
from utils import ensure_parent, get_logger

_log = get_logger("abstention")


class AbstentionProbe(BaseModel):
    """A choice question tagged with whether it is actually answerable."""

    question: Question
    answerable: bool


def unanswerable_from(q: Question, feature_text: str) -> Question:
    """Make an under-determined variant of *q* by removing a decisive feature.

    Removing the discriminating 四诊 cue makes the item multi-answer, so the
    correct behaviour becomes *abstain*; gold is cleared.
    """
    stem = q.stem.replace(feature_text, "")
    return q.model_copy(
        update={"question_id": str(uuid.uuid4()), "stem": stem, "answer": []}, deep=True
    )


# --------------------------------------------------------------------------- #
# Scoring                                                                       #
# --------------------------------------------------------------------------- #


def score_abstention(records: list[tuple]) -> dict:
    """Score ``(EvalRecord, answerable)`` pairs.

    Positive class = "should abstain" (unanswerable).  ``recall`` is **A@D**:
    of the genuinely unanswerable items, the fraction the model abstained on.
    """
    tp = fn = fp = tn = 0          # tp: unanswerable & abstained; fn: unanswerable & answered
    ans_total = ans_correct = 0    # fp: answerable & abstained;  tn: answerable & answered
    for rec, answerable in records:
        abstained = bool(rec.refused)
        if not answerable:
            tp += abstained
            fn += not abstained
        else:
            ans_total += 1
            if abstained:
                fp += 1
            else:
                tn += 1
                ans_correct += bool(rec.correct)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(records),
        "n_unanswerable": tp + fn,
        "n_answerable": ans_total,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),          # = A@D
        "abstention_f1": round(f1, 4),
        "over_abstention_rate": round(fp / ans_total, 4) if ans_total else 0.0,
        "answerable_accuracy": round(ans_correct / ans_total, 4) if ans_total else 0.0,
    }


def evaluate_abstention(probes: list[AbstentionProbe], evaluator) -> dict:
    """Run each probe through *evaluator* and score abstention behaviour."""
    records = [(evaluator.eval_one(p.question), p.answerable) for p in probes]
    return score_abstention(records)


# --------------------------------------------------------------------------- #
# Demo probes + orchestration                                                   #
# --------------------------------------------------------------------------- #

_OPTS = {"A": "肝郁气滞证", "B": "肝火上炎证", "C": "脾胃虚寒证", "D": "脾胃湿热证"}


def _q(stem: str, answer: list[str], cat: Category = Category.COMMON_DISEASE) -> Question:
    return Question(
        question_id=str(uuid.uuid4()), source_passage_id="probe", category=cat, topic_id=0,
        type="single_choice", difficulty="advanced", stem=stem, options=dict(_OPTS),
        answer=answer, explanation="",
    )


def demo_probes() -> list[AbstentionProbe]:
    """Answerable items + their missing-premise (舌脉 removed) unanswerable twins."""
    base = _q(
        "患者女，35岁，胸胁胀痛、善太息、情志抑郁，舌淡红苔薄白，脉弦。其证型最宜辨为？",
        ["A"],
    )
    base2 = _q(
        "患者男，48岁，脘腹冷痛、喜温喜按、纳呆便溏，舌淡胖苔白滑，脉沉迟。其证型最宜辨为？",
        ["C"],
    )
    return [
        AbstentionProbe(question=base, answerable=True),
        AbstentionProbe(question=unanswerable_from(base, "，舌淡红苔薄白，脉弦"), answerable=False),
        AbstentionProbe(question=base2, answerable=True),
        AbstentionProbe(question=unanswerable_from(base2, "，舌淡胖苔白滑，脉沉迟"), answerable=False),
    ]


def run(model: str = "mock", cfg: Optional[Config] = None) -> dict:
    """Evaluate abstention (A@D) for *model* over the demo probes → ``results/``."""
    import json

    from m8_evaluate import ModelEvaluator

    cfg = cfg or load_config()
    metrics = evaluate_abstention(demo_probes(), ModelEvaluator(model, cfg))
    out = ensure_parent(cfg.path("paths.results_dir") / f"abstention_{_slug(model)}.json")
    out.write_text(json.dumps({"model": model, "metrics": metrics}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    _log.info("abstention[%s] = %s", model, metrics)
    return metrics


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


if __name__ == "__main__":
    run()
