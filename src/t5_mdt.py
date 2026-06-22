"""T5 — Multi-agent / MDT consultation (collaboration & disagreement resolution).

Several specialty agents (辨证 / 方剂 / 针灸 / 药学 …) each opine on a case; the
panel's decision is aggregated (confidence-weighted majority, or a chair).  The
headline question (framework tier T5) is **group vs individual**: does the panel
*correct* a single agent's error, or *amplify a shared blind spot*?  Also scored:
disagreement, and whether the group misses a red-flag (safety).

Note the documented failure mode — a homogeneous panel (same base model) tends to
amplify the common blind spot rather than improve.  Offline, the mock panel all
votes the same option, so ``group_gain`` is 0 and wrong cases are *amplified* —
exactly that mode.  Diverse panels (e.g. heterogeneous models) are needed for the
group to outperform the individual.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Callable, Optional

from pydantic import BaseModel, Field

from config import Config, load_config
from llm_client import call_json
from schemas import Category
from utils import ensure_parent, get_logger, resolve_path

_log = get_logger("t5_mdt")

# A specialty agent: case -> SpecialtyOpinion.
SpecialtyAgent = Callable[["MDTCase"], "SpecialtyOpinion"]
Aggregator = Callable[["list[SpecialtyOpinion]"], "list[str]"]

DEFAULT_SPECIALTIES = ["辨证", "方剂", "针灸", "药学"]


# --------------------------------------------------------------------------- #
# Data models                                                                   #
# --------------------------------------------------------------------------- #


class MDTCase(BaseModel):
    case_id: str
    category: Category
    stem: str
    options: dict[str, str]
    answer: list[str]
    red_flag: Optional[str] = None      # a safety keyword the panel must raise


class SpecialtyOpinion(BaseModel):
    specialty: str
    vote: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    rationale: str = ""


class MDTResult(BaseModel):
    case_id: str
    opinions: list[SpecialtyOpinion] = Field(default_factory=list)
    group_vote: list[str] = Field(default_factory=list)
    group_correct: bool = False
    individual_correct: dict[str, bool] = Field(default_factory=dict)
    disagreement: bool = False
    corrected: bool = False              # group right, ≥1 individual wrong
    amplified: bool = False              # group wrong, shared by a majority (blind spot)
    red_flag_present: bool = False
    red_flag_raised: bool = False


# --------------------------------------------------------------------------- #
# Agents + aggregators                                                          #
# --------------------------------------------------------------------------- #


def scripted_agent(specialty: str, vote, confidence: float = 0.8, rationale: str = "") -> SpecialtyAgent:
    """A fixed-opinion specialty agent (for tests / baselines)."""
    votes = vote if isinstance(vote, list) else [vote]

    def _agent(_case: "MDTCase") -> SpecialtyOpinion:
        return SpecialtyOpinion(specialty=specialty, vote=list(votes), confidence=confidence,
                                rationale=rationale)

    return _agent


def llm_specialty_agent(model: str, specialty: str, cfg: Optional[Config] = None) -> SpecialtyAgent:
    """A specialty agent backed by *model* via the MDT prompt."""
    cfg = cfg or load_config()
    try:
        tmpl = resolve_path(cfg.get("prompts.mdt_agent")).read_text(encoding="utf-8")
    except Exception:
        tmpl = "多学科会诊。{specialty}\n{stem}\n选项\n{options}\nJSON {{\"vote\":[\"A\"]}}"

    def _agent(case: "MDTCase") -> SpecialtyOpinion:
        opts = "\n".join(f"{k}. {v}" for k, v in case.options.items())
        prompt = tmpl.format(specialty=specialty, stem=case.stem, options=opts)
        try:
            data = call_json(prompt, model=model)
        except Exception:  # noqa: BLE001
            return SpecialtyOpinion(specialty=specialty, vote=[], confidence=0.0)
        vote = data.get("vote") or []
        if isinstance(vote, str):
            vote = [vote]
        try:
            conf = float(data.get("confidence", 0.5))
        except Exception:  # noqa: BLE001
            conf = 0.5
        return SpecialtyOpinion(specialty=specialty, vote=[str(v) for v in vote],
                                confidence=conf, rationale=str(data.get("rationale", "")))

    return _agent


def weighted_majority(opinions: list[SpecialtyOpinion]) -> list[str]:
    """Confidence-weighted majority vote; deterministic tie-break."""
    score: dict[tuple, float] = defaultdict(float)
    for op in opinions:
        if op.vote:
            score[tuple(sorted(op.vote))] += max(op.confidence, 1e-6)
    if not score:
        return []
    best = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return list(best)


def chair_adjudicator(opinions: list[SpecialtyOpinion]) -> list[str]:
    """A 'chair' that defers to the single highest-confidence specialist."""
    voting = [op for op in opinions if op.vote]
    if not voting:
        return []
    return list(sorted(max(voting, key=lambda o: o.confidence).vote))


# --------------------------------------------------------------------------- #
# Running + scoring                                                             #
# --------------------------------------------------------------------------- #


def run_mdt_case(case: MDTCase, agents: list[SpecialtyAgent],
                 aggregator: Optional[Aggregator] = None) -> MDTResult:
    """Collect specialty opinions, aggregate, and score group vs individual."""
    aggregator = aggregator or weighted_majority
    opinions = [agent(case) for agent in agents]
    gold = sorted(case.answer)
    group_vote = sorted(aggregator(opinions))
    group_correct = group_vote == gold

    individual = {op.specialty: (sorted(op.vote) == gold) for op in opinions}
    votes = [tuple(sorted(op.vote)) for op in opinions]
    disagreement = len(set(votes)) > 1
    corrected = group_correct and not all(individual.values())
    shared = sum(1 for v in votes if v == tuple(group_vote))
    amplified = (not group_correct) and (shared * 2 >= len(agents)) and len(agents) > 0

    red_present = bool(case.red_flag)
    red_raised = red_present and any(case.red_flag in op.rationale for op in opinions)

    return MDTResult(
        case_id=case.case_id, opinions=opinions, group_vote=group_vote,
        group_correct=group_correct, individual_correct=individual, disagreement=disagreement,
        corrected=corrected, amplified=amplified, red_flag_present=red_present,
        red_flag_raised=red_raised,
    )


def aggregate(results: list[MDTResult]) -> dict:
    """Group-vs-individual summary across cases."""
    n = len(results) or 1
    specialties: list[str] = []
    for r in results:
        for s in r.individual_correct:
            if s not in specialties:
                specialties.append(s)
    per_spec = {
        s: sum(r.individual_correct.get(s, False) for r in results) / n for s in specialties
    }
    mean_ind = sum(per_spec.values()) / (len(per_spec) or 1)
    best_ind = max(per_spec.values()) if per_spec else 0.0
    mdt_acc = sum(r.group_correct for r in results) / n
    rf_total = sum(r.red_flag_present for r in results)
    rf_raised = sum(r.red_flag_raised for r in results)
    return {
        "n": len(results),
        "mdt_accuracy": round(mdt_acc, 4),
        "mean_individual_accuracy": round(mean_ind, 4),
        "best_individual_accuracy": round(best_ind, 4),
        "group_gain": round(mdt_acc - mean_ind, 4),            # >0 group helps; <0 amplifies
        "per_specialty_accuracy": {s: round(v, 4) for s, v in per_spec.items()},
        "disagreement_rate": round(sum(r.disagreement for r in results) / n, 4),
        "corrected_rate": round(sum(r.corrected for r in results) / n, 4),
        "amplified_rate": round(sum(r.amplified for r in results) / n, 4),
        "red_flag_recall": round(rf_raised / rf_total, 4) if rf_total else None,
    }


# --------------------------------------------------------------------------- #
# Demo cases + orchestration                                                    #
# --------------------------------------------------------------------------- #


def demo_cases() -> list[MDTCase]:
    return [
        MDTCase(
            case_id="mdt-ganyu", category=Category.COMMON_DISEASE,
            stem="女，35岁，胸胁胀痛、善太息、情志抑郁，舌淡红苔薄白，脉弦。证属？",
            options={"A": "肝郁气滞证", "B": "肝火上炎证", "C": "脾胃虚弱证", "D": "气血两虚证"},
            answer=["A"],
        ),
        MDTCase(
            case_id="mdt-lire", category=Category.COMMON_DISEASE,
            stem="男，40岁，壮热、口渴引饮、面赤、舌红苔黄、脉洪数。证属？",
            options={"A": "风寒表证", "B": "里热实证", "C": "脾胃虚寒证", "D": "肝郁气滞证"},
            answer=["B"],
        ),
        MDTCase(
            case_id="mdt-zhenxintong", category=Category.COMMON_DISEASE,
            stem="男，62岁，突发胸痛彻背、冷汗淋漓、四肢厥冷、脉微欲绝。最宜辨为？",
            options={"A": "真心痛", "B": "胸痹轻证", "C": "肝郁气滞证", "D": "胃脘痛"},
            answer=["A"], red_flag="真心痛",
        ),
    ]


def llm_panel(model: str, specialties: Optional[list[str]] = None,
              cfg: Optional[Config] = None) -> list[SpecialtyAgent]:
    cfg = cfg or load_config()
    return [llm_specialty_agent(model, s, cfg) for s in (specialties or DEFAULT_SPECIALTIES)]


def run(model: str = "mock", cfg: Optional[Config] = None) -> dict:
    """Run the MDT panel for *model* over the demo cases → ``results/``."""
    cfg = cfg or load_config()
    agents = llm_panel(model, cfg=cfg)
    results = [run_mdt_case(c, agents) for c in demo_cases()]
    metrics = aggregate(results)
    payload = {"model": model, "metrics": metrics, "cases": [r.model_dump() for r in results]}
    out = ensure_parent(cfg.path("paths.results_dir") / f"mdt_{_slug(model)}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("t5[%s] = %s", model, metrics)
    return payload


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


if __name__ == "__main__":
    run()
