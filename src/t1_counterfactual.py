"""T1 — Counterfactual minimal pairs + information staging (extends M5).

Two static-format tests from the clinical-eval framework
(``docs/CLINICAL_EVAL_FRAMEWORK.md``, tier T1):

* **Counterfactual minimal pairs.** A base vignette and a variant that differs by
  exactly one 四诊 feature (舌淡↔舌红), with the correct answer *flipped*.  A model
  that does not truly use the feature cannot get both right — so the headline
  metric is **pair accuracy (both correct)**, not single-item accuracy.
* **Information staging.** A vignette split into ordered info units revealed
  cumulatively; the **information efficiency** = the earliest stage at which the
  model first answers correctly (rewards reaching the right call with less info,
  penalises premature wrong commitment).
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from config import Config, load_config
from llm_client import call_json
from schemas import Category, Passage, Question
from utils import ensure_parent, get_logger, load_jsonl_as, resolve_path, save_jsonl

_log = get_logger("t1_counterfactual")


# --------------------------------------------------------------------------- #
# Counterfactual minimal pairs                                                  #
# --------------------------------------------------------------------------- #


class CounterfactualPair(BaseModel):
    """A base question + a minimally-edited variant whose answer flips."""

    pair_id: str
    category: Category
    topic_id: int
    cf_feature: str                 # the decisive feature, e.g. "舌脉"
    base_value: str
    cf_value: str
    base: Question
    variant: Question

    def is_valid(self) -> bool:
        """The pair only makes sense if the two correct answers differ."""
        return bool(self.base.answer) and self.base.answer != self.variant.answer


def _mk_question(stem, options, answer, category, topic_id, feature) -> Question:
    return Question(
        question_id=str(uuid.uuid4()),
        source_passage_id="t1-cf",
        category=category,
        topic_id=int(topic_id),
        type="single_choice",
        difficulty="advanced",
        stem=str(stem).strip(),
        options={str(k): str(v) for k, v in (options or {}).items()},
        answer=[str(a) for a in (answer or [])],
        explanation=f"反事实关键特征：{feature}",
    )


def generate_counterfactual(
    passage: Passage, model: str = "gpt-4o", max_chars: int = 1500
) -> Optional[CounterfactualPair]:
    """Generate one counterfactual minimal pair from a labelled passage."""
    if passage.category is None or passage.topic_id is None:
        return None
    cfg = load_config()
    tmpl = resolve_path(cfg.get("prompts.gen_counterfactual")).read_text(encoding="utf-8")
    prompt = tmpl.format(passage_text=passage.text[:max_chars])
    try:
        d = call_json(prompt, model=model)
    except Exception as exc:  # noqa: BLE001
        _log.warning("counterfactual generation failed: %s", exc)
        return None
    opts = d.get("options") or {}
    base = _mk_question(d["base_stem"], opts, d.get("base_answer"), passage.category,
                        passage.topic_id, d.get("cf_feature", ""))
    variant = _mk_question(d["variant_stem"], opts, d.get("cf_answer"), passage.category,
                           passage.topic_id, d.get("cf_feature", ""))
    pair = CounterfactualPair(
        pair_id=str(uuid.uuid4()),
        category=passage.category,
        topic_id=int(passage.topic_id),
        cf_feature=str(d.get("cf_feature", "")),
        base_value=str(d.get("base_value", "")),
        cf_value=str(d.get("cf_value", "")),
        base=base,
        variant=variant,
    )
    return pair if pair.is_valid() else None


def score_counterfactual_pairs(pairs: list[CounterfactualPair], evaluator) -> dict:
    """Evaluate both items of each pair; report pair accuracy + feature responsiveness.

    * ``pair_accuracy`` — fraction where **both** base and variant are correct
      (the real signal: the model used the flipped feature).
    * ``flip_rate`` — fraction where the model's answer changed between base and
      variant (responded to the feature at all).
    * ``base_accuracy`` / ``variant_accuracy`` — per-side accuracy for context.
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    both = flipped = base_ok = var_ok = 0
    for p in pairs:
        rb = evaluator.eval_one(p.base)
        rv = evaluator.eval_one(p.variant)
        base_ok += rb.correct
        var_ok += rv.correct
        both += rb.correct and rv.correct
        if rb.pred and rv.pred and rb.pred != rv.pred:
            flipped += 1
    return {
        "n": n,
        "pair_accuracy": round(both / n, 4),
        "flip_rate": round(flipped / n, 4),
        "base_accuracy": round(base_ok / n, 4),
        "variant_accuracy": round(var_ok / n, 4),
    }


# --------------------------------------------------------------------------- #
# Information staging                                                            #
# --------------------------------------------------------------------------- #


class StagedCase(BaseModel):
    """A clinical question split into ordered, cumulatively-revealed info units."""

    case_id: str
    category: Category
    topic_id: int
    stages: list[str]                       # info units, least → most informative
    options: dict[str, str]
    answer: list[str]
    explanation: str = ""


_SENT_SPLIT = re.compile(r"(?<=[。；;！？])|，")


def staged_from_question(q: Question, n_stages: int = 4) -> StagedCase:
    """Split a clinical question's stem into ``n_stages`` cumulative info units.

    Heuristic: segment the stem on clause boundaries and chunk into ordered
    stages, so the decisive (last-mentioned) features land in the final stage.
    """
    parts = [s.strip() for s in _SENT_SPLIT.split(q.stem) if s and s.strip()]
    if not parts:
        parts = [q.stem]
    k = max(1, min(n_stages, len(parts)))
    # Distribute clauses across k ordered stages (front-loaded remainder).
    per = len(parts) // k
    extra = len(parts) % k
    stages, i = [], 0
    for s in range(k):
        take = per + (1 if s < extra else 0)
        chunk = parts[i : i + take] or parts[i : i + 1]
        stages.append("，".join(chunk))
        i += take
    return StagedCase(
        case_id=str(uuid.uuid4()),
        category=q.category,
        topic_id=q.topic_id,
        stages=[s for s in stages if s],
        options=dict(q.options),
        answer=list(q.answer),
        explanation=q.explanation,
    )


def evaluate_staging(case: StagedCase, evaluator) -> dict:
    """Reveal stages cumulatively; find the earliest stage answered correctly.

    Returns ``min_correct_stage`` (1-indexed, ``None`` if never), the
    ``information_efficiency`` (1 − (min_stage−1)/total; higher = earlier), and
    whether the model committed a *wrong* answer before its first correct one
    (``early_wrong``).
    """
    total = len(case.stages)
    min_correct: Optional[int] = None
    early_wrong = False
    for k in range(1, total + 1):
        stem = "。".join(case.stages[:k])
        probe = Question(
            question_id=str(uuid.uuid4()),
            source_passage_id=case.case_id,
            category=case.category,
            topic_id=case.topic_id,
            type="single_choice",
            difficulty="advanced",
            stem=stem,
            options=dict(case.options),
            answer=list(case.answer),
            explanation="",
        )
        rec = evaluator.eval_one(probe)
        if rec.correct:
            min_correct = k
            break
        if not rec.refused and rec.pred:
            early_wrong = True
    eff = 0.0 if min_correct is None else round(1 - (min_correct - 1) / max(1, total), 4)
    return {
        "total_stages": total,
        "min_correct_stage": min_correct,
        "information_efficiency": eff,
        "early_wrong": early_wrong,
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def run(model: str = "gpt-4o", limit: Optional[int] = None, cfg: Optional[Config] = None) -> dict:
    """Generate counterfactual pairs from labelled passages → ``interim`` + summary."""
    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    gen_model = cfg.get("generate.model", model)
    passages = load_jsonl_as(interim / "passages_labeled.jsonl", Passage)
    passages = [p for p in passages if p.category is not None and p.topic_id is not None]
    if limit:
        passages = passages[:limit]

    pairs = [pr for p in passages if (pr := generate_counterfactual(p, model=gen_model)) is not None]
    out = ensure_parent(interim / "counterfactual_pairs.jsonl")
    save_jsonl(pairs, out)
    _log.info("generated %d counterfactual pairs from %d passages", len(pairs), len(passages))
    return {"pairs": len(pairs), "path": str(out)}


if __name__ == "__main__":
    run()
