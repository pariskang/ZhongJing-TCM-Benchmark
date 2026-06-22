"""T4 — Longitudinal episode (follow-up & adjustment; trajectory consistency).

A patient across several visits.  The crucial mechanic — what makes this a
*trajectory* rather than independent snapshots — is **outcome-dependent
evolution**: the syndrome at the next visit depends on the treatment chosen now.
Treat correctly → the condition resolves (or advances favourably); treat wrongly
→ it worsens / transitions adversely (表寒入里化热 …).  This lifts 同病异治 from a
single valid-set to a *manifold over the trajectory*: as the syndrome evolves the
plan must adjust.

Scoring (local **and** trajectory, reusing M8 per visit and the L2 decoupling):
per-visit ``node_accuracy``; ``resolved`` / ``failed``; ``adverse_transitions``
(wrong → worsen mid-course); ``adjustment_recall`` (at syndrome-change points,
did the model change its treatment?); and ``clean_resolution`` = cured *without*
ever harming the patient en route (the decoupled "good trajectory").
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Callable, Optional

from pydantic import BaseModel, Field

from config import Config, load_config
from schemas import Category, Question
from utils import ensure_parent, get_logger

_log = get_logger("t4_longitudinal")

# A visit expert: a Question (treatment choice) -> {"pred": [...], "refused": bool}.
VisitExpert = Callable[[Question], dict]


# --------------------------------------------------------------------------- #
# Data models                                                                   #
# --------------------------------------------------------------------------- #


class EpisodeState(BaseModel):
    """One visit's true syndrome, presentation and treatment-choice MCQ + transitions."""

    state_id: str
    syndrome: str                       # current true syndrome (hidden answer)
    presentation: str                   # what the patient presents this visit
    options: dict[str, str]             # treatment (治法/方) options
    answer: list[str]                   # correct option key(s) for this syndrome
    on_correct: Optional[str] = None    # next state if treated correctly (None = resolved)
    on_wrong: Optional[str] = None      # next state if treated wrongly (None = terminal failure)


class LongitudinalEpisode(BaseModel):
    episode_id: str
    category: Category
    start: str
    states: dict[str, EpisodeState]
    max_visits: int = 4


class VisitResult(BaseModel):
    visit: int
    state_id: str
    syndrome: str
    chosen: list[str] = Field(default_factory=list)
    chosen_treatment: list[str] = Field(default_factory=list)
    correct: bool = False
    refused: bool = False


class EpisodeResult(BaseModel):
    episode_id: str
    model: str
    visits: list[VisitResult] = Field(default_factory=list)
    node_accuracy: float = 0.0
    resolved: bool = False
    failed: bool = False
    adverse_transitions: int = 0
    syndrome_changes: int = 0
    adjusted: int = 0
    adjustment_recall: float = 1.0
    clean_resolution: bool = False


# --------------------------------------------------------------------------- #
# Experts                                                                       #
# --------------------------------------------------------------------------- #


def mcq_expert(model: str, cfg: Optional[Config] = None) -> VisitExpert:
    """Per-visit treatment decision via the M8 STAGER evaluator (reuses M8)."""
    from m8_evaluate import ModelEvaluator

    ev = ModelEvaluator(model, cfg or load_config())

    def _expert(q: Question) -> dict:
        rec = ev.eval_one(q)
        return {"pred": rec.pred, "refused": rec.refused}

    return _expert


def scripted_visit_expert(picks: list) -> VisitExpert:
    """Replay fixed per-visit picks (a key, a list of keys, or ``"REFUSE"``)."""
    it = iter(picks)

    def _expert(_q: Question) -> dict:
        try:
            p = next(it)
        except StopIteration:
            return {"pred": [], "refused": True}
        if p == "REFUSE":
            return {"pred": [], "refused": True}
        return {"pred": p if isinstance(p, list) else [p], "refused": False}

    return _expert


# --------------------------------------------------------------------------- #
# Episode loop + scoring                                                         #
# --------------------------------------------------------------------------- #


def _visit_question(state: EpisodeState, category: Category) -> Question:
    return Question(
        question_id=str(uuid.uuid4()), source_passage_id="t4", category=category, topic_id=0,
        type="single_choice", difficulty="advanced", stem=state.presentation,
        options=dict(state.options), answer=list(state.answer), explanation="",
    )


def run_episode(episode: LongitudinalEpisode, expert: VisitExpert,
                max_visits: Optional[int] = None, model: str = "?") -> EpisodeResult:
    """Drive *expert* visit-by-visit; the syndrome evolves with the treatment chosen."""
    max_visits = max_visits or episode.max_visits
    state = episode.states[episode.start]
    visits: list[VisitResult] = []
    prev_syndrome: Optional[str] = None
    prev_treatment: Optional[list[str]] = None
    adverse = syndrome_changes = adjusted = 0
    resolved = failed = False

    for v in range(max_visits):
        out = expert(_visit_question(state, episode.category))
        pred = sorted(out.get("pred", []))
        refused = bool(out.get("refused"))
        correct = (not refused) and pred == sorted(state.answer)
        treatment = [state.options[k] for k in pred if k in state.options]

        if prev_syndrome is not None and state.syndrome != prev_syndrome:
            syndrome_changes += 1
            if treatment != prev_treatment:          # adjusted the plan to the new syndrome
                adjusted += 1

        visits.append(VisitResult(
            visit=v + 1, state_id=state.state_id, syndrome=state.syndrome,
            chosen=pred, chosen_treatment=treatment, correct=correct, refused=refused,
        ))
        prev_syndrome, prev_treatment = state.syndrome, treatment

        nxt = state.on_correct if correct else state.on_wrong
        if nxt is None:
            resolved, failed = correct, not correct
            break
        if not correct:
            adverse += 1                              # worsened and continued
        state = episode.states[nxt]

    node_acc = sum(x.correct for x in visits) / len(visits) if visits else 0.0
    recall = round(adjusted / syndrome_changes, 4) if syndrome_changes else 1.0
    return EpisodeResult(
        episode_id=episode.episode_id, model=model, visits=visits,
        node_accuracy=round(node_acc, 4), resolved=resolved, failed=failed,
        adverse_transitions=adverse, syndrome_changes=syndrome_changes, adjusted=adjusted,
        adjustment_recall=recall, clean_resolution=resolved and adverse == 0,
    )


def aggregate(results: list[EpisodeResult]) -> dict:
    n = len(results) or 1
    changes = sum(r.syndrome_changes for r in results)
    return {
        "n": len(results),
        "node_accuracy": round(sum(r.node_accuracy for r in results) / n, 4),
        "resolution_rate": round(sum(r.resolved for r in results) / n, 4),
        "clean_resolution_rate": round(sum(r.clean_resolution for r in results) / n, 4),
        "failure_rate": round(sum(r.failed for r in results) / n, 4),
        "mean_adverse_transitions": round(sum(r.adverse_transitions for r in results) / n, 4),
        "adjustment_recall": round(sum(r.adjusted for r in results) / changes, 4) if changes else 1.0,
    }


# --------------------------------------------------------------------------- #
# Demo episodes + orchestration                                                 #
# --------------------------------------------------------------------------- #


def demo_episodes() -> list[LongitudinalEpisode]:
    """Two 外感 trajectories (treat-right resolves; treat-wrong drives 入里化热)."""
    cat = Category.COMMON_DISEASE
    smooth = LongitudinalEpisode(
        episode_id="ep-windcold-smooth", category=cat, start="s0", max_visits=4,
        states={
            "s0": EpisodeState(
                state_id="s0", syndrome="风寒表证",
                presentation="初诊：恶寒发热、无汗、头身疼痛，舌淡苔薄白，脉浮紧。当用何治法？",
                options={"A": "辛温解表", "B": "辛凉解表", "C": "清热泻火", "D": "温里散寒"},
                answer=["A"], on_correct=None, on_wrong="s1"),
            "s1": EpisodeState(
                state_id="s1", syndrome="表邪入里化热",
                presentation="复诊：服药后高热、口渴引饮、心烦，舌红苔黄，脉数。现当用何治法？",
                options={"A": "清热泻火", "B": "辛温解表", "C": "辛凉解表", "D": "温里散寒"},
                answer=["A"], on_correct=None, on_wrong="s2"),
            "s2": EpisodeState(
                state_id="s2", syndrome="热盛伤津变证",
                presentation="复诊：壮热不退、津伤口干、舌绛少津。",
                options={"A": "清热生津", "B": "辛温解表", "C": "温里散寒", "D": "消食导滞"},
                answer=["A"], on_correct=None, on_wrong=None),
        },
    )
    misled = LongitudinalEpisode(
        episode_id="ep-windcold-misled", category=cat, start="t0", max_visits=4,
        states={
            "t0": EpisodeState(
                state_id="t0", syndrome="风寒表证",
                presentation="初诊：恶寒重发热轻、无汗、骨节酸痛，舌淡苔白，脉浮紧。当用何治法？",
                options={"A": "辛凉解表", "B": "辛温解表", "C": "清热解毒", "D": "消食导滞"},
                answer=["B"], on_correct=None, on_wrong="t1"),
            "t1": EpisodeState(
                state_id="t1", syndrome="表邪入里化热",
                presentation="复诊：误治后高热、口渴、舌红苔黄，脉滑数。现当用何治法？",
                options={"A": "清热泻火", "B": "辛温解表", "C": "温里散寒", "D": "消食导滞"},
                answer=["A"], on_correct=None, on_wrong="t2"),
            "t2": EpisodeState(
                state_id="t2", syndrome="热入营血变证",
                presentation="复诊：身热夜甚、斑疹隐隐、舌绛。",
                options={"A": "清营凉血", "B": "辛温解表", "C": "温里散寒", "D": "消食导滞"},
                answer=["A"], on_correct=None, on_wrong=None),
        },
    )
    return [smooth, misled]


def run(model: str = "mock", cfg: Optional[Config] = None) -> dict:
    """Run T4 longitudinal episodes for *model* over the demo set → ``results/``."""
    cfg = cfg or load_config()
    expert = mcq_expert(model, cfg)
    results = [run_episode(ep, expert, model=model) for ep in demo_episodes()]
    metrics = aggregate(results)
    payload = {"model": model, "metrics": metrics, "episodes": [r.model_dump() for r in results]}
    out = ensure_parent(cfg.path("paths.results_dir") / f"episode_{_slug(model)}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("t4[%s] = %s", model, metrics)
    return payload


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


if __name__ == "__main__":
    run()
