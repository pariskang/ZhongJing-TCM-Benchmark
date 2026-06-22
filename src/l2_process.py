"""L2 — Process gate + step-level PRM data (clinical-eval framework §4, §9).

Trajectory-level signal, not just the final answer:

* **Step-PRM data.** Each step case = a clinical *context* + a **correct** next
  action + a **plausible-but-wrong** action (premature closure / redundancy) +
  an optional **neutral** harmless-useless action (negative calibration, since
  process judges over-reward "active" steps).  Built deterministically from the
  T2 cases.
* **Process preference.** A judge (the model under test) must prefer the correct
  action over the wrong one; reported as ``process_preference_accuracy`` (and a
  correct-vs-neutral trap rate).
* **Process gate (result/process decoupling).** A "right answer" reached by an
  unsound process (e.g. premature closure in a T2 consultation) is **downgraded**
  — ``gated_accuracy`` ≤ raw ``accuracy``.
"""
from __future__ import annotations

import json
import random
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from config import Config, load_config
from llm_client import call_json
from utils import ensure_parent, get_logger, resolve_path

_log = get_logger("l2_process")

# A process judge: (context, action_a, action_b) -> "A" | "B".
ProcessJudge = Callable[[str, str, str], str]

_GOOD = ("追问", "采集", "判别", "四诊", "辨证", "据此")
_BAD = ("立即", "尚未", "未问", "直接判定", "已问", "已采集", "无关", "寒暄", "重复")


def _soundness(text: str) -> int:
    """Heuristic action soundness (used by the baseline judge and the mock)."""
    return sum(g in text for g in _GOOD) - sum(b in text for b in _BAD)


# --------------------------------------------------------------------------- #
# Step-PRM data                                                                 #
# --------------------------------------------------------------------------- #


class StepCase(BaseModel):
    """One step-level preference item (correct vs plausible-wrong vs neutral)."""

    step_id: str
    context: str
    correct_action: str
    wrong_action: str               # plausible but wrong (premature / redundant)
    neutral_action: Optional[str] = None  # harmless but useless (calibration)
    rationale: str = ""


def build_step_cases(case) -> list[StepCase]:
    """Two decision points per :class:`~t2_patient_sim.ClinicalCase`.

    Early (under-informed) → must acquire discriminating 四诊, not diagnose;
    Sufficient (key features gathered) → must diagnose, not keep probing.
    """
    opening = f"{case.demographics}。{case.chief_complaint}"
    key_facts = "；".join(f"{k}：{case.findings[k]}" for k in case.key_features if k in case.findings)
    decoy = case.differentials[0] if case.differentials else "某证"
    return [
        StepCase(
            step_id=f"{case.case_id}-early",
            context=f"问诊刚开始，仅知：{opening}。判别性四诊（舌、脉等）尚未采集。",
            correct_action="追问舌象、脉象等判别性四诊以助鉴别",
            wrong_action=f"尚未问舌脉即直接判定为{decoy}",
            neutral_action="询问与本病无关的多年前阑尾手术史",
            rationale="信息不足时应先采集判别性信息,不可过早闭合。",
        ),
        StepCase(
            step_id=f"{case.case_id}-sufficient",
            context=f"已采集：{opening}；{key_facts}。判别性四诊已齐。",
            correct_action="据此四诊辨证并给出证型诊断",
            wrong_action="继续重复追问已问过的舌象",
            neutral_action="和患者寒暄聊聊天气",
            rationale="信息足够即应决策,冗余追问无信息增益。",
        ),
    ]


# --------------------------------------------------------------------------- #
# Process judges + preference scoring                                           #
# --------------------------------------------------------------------------- #


def baseline_process_judge(context: str, action_a: str, action_b: str) -> str:
    """Content-based judge (position-independent): prefer the sounder action."""
    return "A" if _soundness(action_a) >= _soundness(action_b) else "B"


def llm_process_judge(model: str, cfg: Optional[Config] = None) -> ProcessJudge:
    """A process judge backed by *model* via the step-PRM prompt."""
    cfg = cfg or load_config()
    try:
        tmpl = resolve_path(cfg.get("prompts.prm_step")).read_text(encoding="utf-8")
    except Exception:
        tmpl = "情境:{context}\n候选A: {action_a}\n候选B: {action_b}\n输出 {{\"better\":\"A\"}}"

    def _judge(context: str, action_a: str, action_b: str) -> str:
        prompt = tmpl.format(context=context, action_a=action_a, action_b=action_b)
        try:
            data = call_json(prompt, model=model)
        except Exception:  # noqa: BLE001
            return "A"
        return "B" if str(data.get("better", "A")).strip().upper().startswith("B") else "A"

    return _judge


def score_process(step_cases: list[StepCase], judge: ProcessJudge, seed: int = 0) -> dict:
    """Preference accuracy: does *judge* pick correct over wrong (and over neutral)?

    Correct/wrong are presented in a randomised A/B order (per step) to neutralise
    position bias, then mapped back to the semantic label.
    """
    n = len(step_cases)
    if n == 0:
        return {"n": 0}
    correct_pref = 0
    neutral_total = neutral_pref = 0
    for sc in step_cases:
        order = [("correct", sc.correct_action), ("wrong", sc.wrong_action)]
        random.Random(f"{sc.step_id}|{seed}").shuffle(order)
        pick = judge(sc.context, order[0][1], order[1][1])
        chosen = order[0][0] if pick == "A" else order[1][0]
        correct_pref += chosen == "correct"
        if sc.neutral_action:
            neutral_total += 1
            order2 = [("correct", sc.correct_action), ("neutral", sc.neutral_action)]
            random.Random(f"{sc.step_id}|n|{seed}").shuffle(order2)
            pick2 = judge(sc.context, order2[0][1], order2[1][1])
            chosen2 = order2[0][0] if pick2 == "A" else order2[1][0]
            neutral_pref += chosen2 == "correct"
    out = {"n": n, "process_preference_accuracy": round(correct_pref / n, 4)}
    if neutral_total:
        out["correct_vs_neutral_accuracy"] = round(neutral_pref / neutral_total, 4)
    return out


# --------------------------------------------------------------------------- #
# Process gate (result/process decoupling)                                      #
# --------------------------------------------------------------------------- #


def process_gate(l1_correct: bool, l2_sound: bool) -> dict:
    """A question/episode passes only if the result is correct **and** sound."""
    return {"l1_correct": bool(l1_correct), "l2_sound": bool(l2_sound),
            "gated_pass": bool(l1_correct and l2_sound)}


def gate_consultations(results) -> dict:
    """Apply the process gate to T2 consultations: a premature-closure correct is downgraded."""
    n = len(results) or 1
    raw = sum(r.correct for r in results)
    gated = sum(r.correct and not r.premature_closure for r in results)
    return {
        "n": len(results),
        "raw_accuracy": round(raw / n, 4),
        "gated_accuracy": round(gated / n, 4),
        "downgraded": raw - gated,   # right answer, unsound process
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def run(model: str = "mock", cfg: Optional[Config] = None) -> dict:
    """Score step-PRM preference + the consultation process gate for *model*."""
    cfg = cfg or load_config()
    from t2_patient_sim import demo_cases, evaluate_consultation, llm_expert

    cases = demo_cases()
    step_cases = [sc for c in cases for sc in build_step_cases(c)]
    process = score_process(step_cases, llm_process_judge(model, cfg))
    _, results = evaluate_consultation(cases, llm_expert(model, cfg), model=model, cfg=cfg)
    gate = gate_consultations(results)

    payload = {"model": model, "process": process, "gate": gate, "n_step_cases": len(step_cases)}
    out = ensure_parent(cfg.path("paths.results_dir") / f"l2_process_{_slug(model)}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("l2[%s] process=%s gate=%s", model, process, gate)
    return payload


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


if __name__ == "__main__":
    run()
