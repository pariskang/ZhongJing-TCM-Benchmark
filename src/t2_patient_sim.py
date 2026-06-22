"""T2 — Active-inquiry consultation against a patient simulator (POMDP).

Lifts evaluation from "answer a complete-information MCQ" to "reveal the diagnosis
by *asking*", the core of the clinical-eval framework
(``docs/CLINICAL_EVAL_FRAMEWORK.md``, tier T2).

Pieces
------
* :class:`ClinicalCase` — a vignette with a **hidden** syndrome and symptom-level
  ``findings`` (四诊).  ``key_features`` are the discriminating aspects.
* :class:`PatientSim` — answers only what is asked, with **zero answer leakage**
  (never states a diagnosis/syndrome name); deterministic in ``mock`` mode,
  LLM-backed otherwise.
* :func:`run_consultation` — drives an *expert* (the model under test) over a
  multi-turn ask→answer loop, then scores it.
* Scoring (decoupled result/process, §4 of the framework): final correctness
  (L1) **and** inquiry efficiency / timely closure (L2) — turns used,
  key-feature recall, premature-closure, abstention.

The expert never sees the hidden state — only the running dialogue.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from config import Config, load_config
from llm_client import call_json, call_text
from utils import ensure_parent, get_logger, resolve_path

_log = get_logger("t2_patient_sim")

# An *expert* is a callable: dialogue history -> next action dict.
Action = dict
Expert = Callable[[list[tuple[str, str]]], Action]

#: Doctor queries that fish for the answer rather than a symptom -> deflected.
_DX_QUERY = re.compile(r"诊断|什么病|什么证|证型|辨证|病机|是不是.*证")

#: Aspect → trigger substrings, used to decide which finding a question targets.
_ASPECT_TRIGGERS: dict[str, list[str]] = {
    "主症": ["主要", "部位", "性质", "哪里", "哪儿", "伴随", "症状", "不适"],
    "舌象": ["舌"],
    "脉象": ["脉"],
    "情志": ["情志", "情绪", "心情", "压力", "抑郁", "焦虑"],
    "寒热": ["寒", "热", "怕冷", "发热", "恶寒"],
    "汗": ["汗"],
    "头身": ["头", "身", "肢体", "乏力"],
    "饮食": ["饮食", "纳", "食欲", "胃口", "吃"],
    "口渴": ["渴", "口干", "饮水"],
    "睡眠": ["睡", "眠", "失眠", "梦"],
    "二便": ["大便", "小便", "二便", "便", "尿"],
    "病程": ["多久", "病程", "起病", "多长时间", "什么时候", "诱因", "怎么引起", "如何引起"],
    "月经": ["月经", "经期", "痛经", "带下", "月经", "经"],
}


def _aspects_for_query(query: str, findings: dict[str, str]) -> list[str]:
    """Which finding aspects a doctor's *query* asks about (present in the case)."""
    hits = []
    for aspect in findings:
        triggers = _ASPECT_TRIGGERS.get(aspect, []) + [aspect]
        if any(t in query for t in triggers):
            hits.append(aspect)
    return hits


# --------------------------------------------------------------------------- #
# Data models                                                                   #
# --------------------------------------------------------------------------- #


class ClinicalCase(BaseModel):
    """A consultation vignette with a hidden syndrome (the POMDP state)."""

    case_id: str
    demographics: str                       # "女，35岁"
    chief_complaint: str                    # opening info shown to the expert
    findings: dict[str, str]                # aspect -> symptom-level fact (四诊)
    hidden_syndrome: str                    # the answer — NEVER revealed by the patient
    key_features: list[str] = Field(default_factory=list)   # discriminating aspects
    differentials: list[str] = Field(default_factory=list)  # confusable syndromes


class ConsultationResult(BaseModel):
    """Scored outcome of one consultation (L1 result + L2 process)."""

    case_id: str
    model: str
    turns_used: int
    asked: list[str] = Field(default_factory=list)
    revealed_aspects: list[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    correct: bool = False
    abstained: bool = False
    premature_closure: bool = False
    key_feature_hits: int = 0
    key_feature_total: int = 0


# --------------------------------------------------------------------------- #
# Patient simulator                                                             #
# --------------------------------------------------------------------------- #

_DEFAULT_PATIENT_TMPL = (
    "你是标准化病人。资料(仅供参考):{demographics}\n{findings}\n"
    "只回答被问到的症状级信息,严禁说出任何诊断/证型名。\n问题:{query}\n回答:"
)


class PatientSim:
    """Answers symptom-level questions about a case with zero answer leakage."""

    def __init__(self, case: ClinicalCase, model: str = "mock", cfg: Optional[Config] = None):
        self.case = case
        self.model = model
        self.cfg = cfg or load_config()
        self._tmpl = self._load_tmpl()

    def _load_tmpl(self) -> str:
        try:
            return resolve_path(self.cfg.get("prompts.patient_sim")).read_text(encoding="utf-8")
        except Exception:
            return _DEFAULT_PATIENT_TMPL

    def _redact(self, text: str) -> str:
        """Strip any diagnosis/syndrome name (the answer or a differential)."""
        for s in [self.case.hidden_syndrome, *self.case.differentials]:
            if not s:
                continue
            text = text.replace(s, "〔已隐去〕")
            base = s.rstrip("证")
            if base and base != s:
                text = text.replace(base, "〔已隐去〕")
        return text

    def answer(self, query: str) -> tuple[str, list[str]]:
        """Return ``(reply, revealed_aspects)``; reply never leaks the diagnosis."""
        if _DX_QUERY.search(query):                       # deflect answer-fishing
            return self._redact("我不懂这些，我只知道自己难受的地方。"), []
        aspects = _aspects_for_query(query, self.case.findings)
        if self.model and self.model.lower() != "mock":
            findings = "\n".join(f"・{k}: {v}" for k, v in self.case.findings.items())
            prompt = self._tmpl.format(demographics=self.case.demographics, findings=findings, query=query)
            reply = call_text(prompt, model=self.model)
        elif aspects:
            reply = "；".join(self.case.findings[a] for a in aspects)
        else:
            reply = "这方面我没有特别不适。"
        return self._redact(reply), aspects


# --------------------------------------------------------------------------- #
# Experts (the model under test)                                                #
# --------------------------------------------------------------------------- #


def _format_history(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{'患者' if role == 'patient' else '医生'}：{text}" for role, text in history)


def scripted_expert(actions: list[Action]) -> Expert:
    """A deterministic expert that replays *actions* (for tests/baselines)."""
    it = iter(actions)

    def _expert(_history: list[tuple[str, str]]) -> Action:
        try:
            return next(it)
        except StopIteration:
            return {"action": "abstain"}

    return _expert


def llm_expert(model: str, cfg: Optional[Config] = None) -> Expert:
    """An expert backed by *model* (the system under test) via the inquiry prompt."""
    cfg = cfg or load_config()
    try:
        tmpl = resolve_path(cfg.get("prompts.expert_inquiry")).read_text(encoding="utf-8")
    except Exception:
        tmpl = "你是接诊医生。\n【对话】\n{dialogue}\n输出JSON行动:"

    def _expert(history: list[tuple[str, str]]) -> Action:
        prompt = tmpl.format(dialogue=_format_history(history))
        try:
            data = call_json(prompt, model=model)
        except Exception as exc:  # noqa: BLE001
            _log.debug("expert parse failed (%s); abstaining", exc)
            return {"action": "abstain"}
        act = str(data.get("action", "")).lower()
        if act.startswith("diag"):
            return {"action": "diagnose", "answer": data.get("answer") or data.get("diagnosis")}
        if act.startswith("abst"):
            return {"action": "abstain"}
        return {"action": "ask", "query": data.get("query") or data.get("question") or ""}

    return _expert


# --------------------------------------------------------------------------- #
# Consultation loop + scoring                                                   #
# --------------------------------------------------------------------------- #


def _norm_dx(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"[\s，,。.；;:：、]", "", s)
    return s.replace("证型", "").rstrip("证")


def run_consultation(
    case: ClinicalCase,
    expert: Expert,
    patient: Optional[PatientSim] = None,
    max_turns: int = 8,
    model: str = "?",
) -> ConsultationResult:
    """Drive *expert* through an ask→answer loop with *patient*, then score it."""
    patient = patient or PatientSim(case, model="mock")
    history: list[tuple[str, str]] = [("patient", f"{case.demographics}。{case.chief_complaint}")]
    asked: list[str] = []
    revealed: set[str] = set()
    final_answer: Optional[str] = None
    abstained = False

    for _ in range(max_turns):
        action = expert(history)
        kind = action.get("action")
        if kind == "diagnose":
            final_answer = action.get("answer")
            break
        if kind == "abstain":
            abstained = True
            break
        query = str(action.get("query", "")).strip()
        if not query:
            abstained = True
            break
        reply, aspects = patient.answer(query)
        revealed.update(aspects)
        asked.append(query)
        history.append(("doctor", query))
        history.append(("patient", reply))

    key_total = len(case.key_features)
    key_hits = sum(1 for k in case.key_features if k in revealed)
    correct = bool(final_answer) and _norm_dx(final_answer) == _norm_dx(case.hidden_syndrome)
    premature = (final_answer is not None) and (key_hits < key_total)
    return ConsultationResult(
        case_id=case.case_id,
        model=model,
        turns_used=len(asked),
        asked=asked,
        revealed_aspects=sorted(revealed),
        final_answer=final_answer,
        correct=correct,
        abstained=abstained,
        premature_closure=premature,
        key_feature_hits=key_hits,
        key_feature_total=key_total,
    )


def evaluate_consultation(
    cases: list[ClinicalCase],
    expert: Expert,
    max_turns: int = 8,
    patient_model: str = "mock",
    model: str = "?",
    cfg: Optional[Config] = None,
) -> tuple[dict, list[ConsultationResult]]:
    """Aggregate consultation metrics over *cases*."""
    cfg = cfg or load_config()
    results = [
        run_consultation(c, expert, PatientSim(c, model=patient_model, cfg=cfg), max_turns, model)
        for c in cases
    ]
    n = len(results) or 1
    key_total = sum(r.key_feature_total for r in results) or 1
    metrics = {
        "model": model,
        "n": len(results),
        "accuracy": round(sum(r.correct for r in results) / n, 4),
        "mean_turns": round(sum(r.turns_used for r in results) / n, 4),
        "premature_closure_rate": round(sum(r.premature_closure for r in results) / n, 4),
        "abstention_rate": round(sum(r.abstained for r in results) / n, 4),
        "key_feature_recall": round(sum(r.key_feature_hits for r in results) / key_total, 4),
    }
    return metrics, results


# --------------------------------------------------------------------------- #
# Demo cases + orchestration                                                    #
# --------------------------------------------------------------------------- #


def demo_cases() -> list[ClinicalCase]:
    """A small built-in case set (consistent with the mock generator's theme)."""
    return [
        ClinicalCase(
            case_id="case-ganyu-qizhi",
            demographics="女，35岁",
            chief_complaint="反复胸胁胀痛、情志抑郁2月",
            findings={
                "主症": "胸胁胀痛、走窜不定，时轻时重，善太息，太息后稍舒",
                "情志": "情志抑郁，遇情绪波动则加重",
                "病程": "起病2月，因家事不顺渐起，无明显寒热",
                "饮食": "纳食一般，时有嗳气、脘腹胀闷",
                "睡眠": "入睡尚可，多梦易醒",
                "二便": "大便不爽，小便正常",
                "舌象": "舌淡红，苔薄白",
                "脉象": "脉弦",
                "月经": "经前乳房胀痛，经行不畅，色暗有块",
            },
            hidden_syndrome="肝郁气滞证",
            key_features=["主症", "舌象", "脉象"],
            differentials=["肝火上炎证", "肝阳上亢证", "心脾两虚证", "痰气郁结证"],
        ),
        ClinicalCase(
            case_id="case-piwei-qixu",
            demographics="男，48岁",
            chief_complaint="食少、神疲乏力半年",
            findings={
                "主症": "食少纳呆，食后脘腹胀满，神疲乏力，少气懒言",
                "病程": "起病半年，劳累后加重，休息可缓",
                "寒热": "无寒热，畏冷不显",
                "饮食": "纳少，喜温食，食凉则腹胀便溏",
                "二便": "大便溏薄，日2-3次",
                "头身": "肢体倦怠，面色萎黄",
                "舌象": "舌淡胖，边有齿痕，苔白",
                "脉象": "脉缓弱",
            },
            hidden_syndrome="脾胃气虚证",
            key_features=["主症", "舌象", "脉象"],
            differentials=["脾阳虚证", "脾胃湿热证", "肝郁脾虚证", "胃阴虚证"],
        ),
    ]


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def run(model: str = "mock", max_turns: int = 8, cfg: Optional[Config] = None) -> dict:
    """Run the T2 consultation eval for *model* over the demo cases → ``results/``."""
    cfg = cfg or load_config()
    cases = demo_cases()
    expert = llm_expert(model, cfg)
    metrics, results = evaluate_consultation(
        cases, expert, max_turns=max_turns, patient_model="mock", model=model, cfg=cfg
    )
    out = ensure_parent(cfg.path("paths.results_dir") / f"consult_{_slug(model)}.json")
    out.write_text(
        json.dumps(
            {"model": model, "metrics": metrics, "consultations": [r.model_dump() for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _log.info("consult[%s] metrics=%s", model, metrics)
    return metrics


if __name__ == "__main__":
    import sys

    run(model=sys.argv[1] if len(sys.argv) > 1 else "mock")
