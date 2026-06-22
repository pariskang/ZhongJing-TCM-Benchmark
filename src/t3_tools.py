"""T3 — Tool-use agent with a tool-grounding contradiction probe.

The framework (tier T3) provides real tools and tests whether the agent (a)
decides to use the right one with correct args, and (b) **grounds its conclusion
in the tool result** — the key diagnostic being a deliberate *tool-result vs
model-claim contradiction* point that catches tool-grounding hallucination
(answering without consulting the tool, or against what it returned).

Tools here are deterministic TCM checkers (十八反/十九畏 compatibility, dose
range).  Scoring is end-to-end (task success) **and** trajectory-level (did it
call the required tool? does the final answer contradict the observed result?).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from config import Config, load_config
from llm_client import call_json
from utils import ensure_parent, get_logger, resolve_path

_log = get_logger("t3_tools")

# A tool agent: (task, history) -> action dict.
ToolAgent = Callable[["ToolTask", list], dict]


# --------------------------------------------------------------------------- #
# Tools (deterministic TCM checkers)                                            #
# --------------------------------------------------------------------------- #

#: 十八反 — mutually-incompatible groups (any member of A with any member of B).
_FAN: list[tuple[list[str], list[str]]] = [
    (["甘草"], ["甘遂", "大戟", "海藻", "芫花"]),
    (["乌头", "川乌", "草乌", "附子"],
     ["半夏", "瓜蒌", "天花粉", "贝母", "川贝", "浙贝", "白蔹", "白及"]),
    (["藜芦"], ["人参", "丹参", "玄参", "沙参", "苦参", "细辛", "芍药"]),
]
#: 十九畏 — incompatible pairs.
_WEI: list[tuple[str, str]] = [
    ("硫黄", "朴硝"), ("水银", "砒霜"), ("狼毒", "密陀僧"), ("巴豆", "牵牛"),
    ("丁香", "郁金"), ("川乌", "犀角"), ("草乌", "犀角"), ("牙硝", "三棱"),
    ("官桂", "石脂"), ("人参", "五灵脂"),
]
#: Coarse safe dose ranges (g) for the dose checker.
_DOSE_MAX = {"附子": 15.0, "细辛": 3.0, "甘草": 10.0, "麻黄": 9.0, "半夏": 9.0}


def _present(term: str, herbs: list[str]) -> Optional[str]:
    for h in herbs:
        if term in h or h in term:
            return h
    return None


def contraindication_check(herbs: list[str]) -> dict:
    """Check a prescription for 十八反 / 十九畏 conflicts."""
    herbs = [str(h).strip() for h in (herbs or []) if str(h).strip()]
    pairs: list[list[str]] = []
    for group_a, group_b in _FAN:
        for ta in group_a:
            ha = _present(ta, herbs)
            if not ha:
                continue
            for tb in group_b:
                hb = _present(tb, herbs)
                if hb:
                    pairs.append(sorted([ha, hb]))
    for x, y in _WEI:
        hx, hy = _present(x, herbs), _present(y, herbs)
        if hx and hy:
            pairs.append(sorted([hx, hy]))
    uniq = sorted({tuple(p) for p in pairs})
    return {"conflict": bool(uniq), "pairs": [list(p) for p in uniq]}


def dose_check(herb: str, dose_g: float) -> dict:
    """Check whether a herb dose exceeds its coarse safe maximum (g)."""
    mx = _DOSE_MAX.get(str(herb).strip())
    try:
        dose = float(dose_g)
    except Exception:  # noqa: BLE001
        return {"error": "invalid dose"}
    if mx is None:
        return {"known": False, "over": False}
    return {"known": True, "over": dose > mx, "max": mx}


# --------------------------------------------------------------------------- #
# Data models + environment                                                     #
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    result: dict = Field(default_factory=dict)


class ToolTask(BaseModel):
    task_id: str
    prompt: str
    required_tool: str
    gold_answer: str
    verdict_key: str                                  # field of the tool result that decides
    verdict_map: dict[str, str] = Field(default_factory=dict)  # "true"/"false" -> answer

    def implied_answer(self, result: dict) -> Optional[str]:
        return self.verdict_map.get(str(bool(result.get(self.verdict_key))).lower())


class ToolEpisodeResult(BaseModel):
    task_id: str
    model: str
    steps: int
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: Optional[str] = None
    correct: bool = False
    called_required: bool = False
    tool_contradiction: bool = False       # final answer contradicts an observed tool result
    grounded: bool = False
    success: bool = False


class ToolEnv:
    """Executes tool calls and exposes the tool specs to the agent."""

    def __init__(self):
        self.tools: dict[str, Callable[..., dict]] = {
            "contraindication_check": contraindication_check,
            "dose_check": dose_check,
        }

    def specs(self) -> str:
        return (
            "- contraindication_check(herbs: list[str]) 检查十八反/十九畏配伍禁忌，返回 {conflict, pairs}\n"
            "- dose_check(herb: str, dose_g: number) 检查剂量是否超量，返回 {over, max}"
        )

    def execute(self, call: ToolCall) -> ToolResult:
        fn = self.tools.get(call.tool)
        if fn is None:
            return ToolResult(tool=call.tool, result={"error": "unknown tool"})
        try:
            return ToolResult(tool=call.tool, result=fn(**call.args))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=call.tool, result={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Agents                                                                        #
# --------------------------------------------------------------------------- #


def _format_history(history: list[tuple[str, dict]]) -> str:
    lines = []
    for role, payload in history:
        tag = "调用工具" if role == "call" else "工具结果"
        lines.append(f"{tag}: {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def scripted_tool_agent(actions: list[dict]) -> ToolAgent:
    """Replay a fixed action list (for tests / baselines)."""
    it = iter(actions)

    def _agent(_task: "ToolTask", _history: list) -> dict:
        try:
            return next(it)
        except StopIteration:
            return {"action": "final", "answer": ""}

    return _agent


def llm_tool_agent(model: str, env: ToolEnv, cfg: Optional[Config] = None) -> ToolAgent:
    """A tool agent backed by *model* via the tool-agent prompt."""
    cfg = cfg or load_config()
    try:
        tmpl = resolve_path(cfg.get("prompts.tool_agent")).read_text(encoding="utf-8")
    except Exception:
        tmpl = "你是中医临床智能体。任务: {task}\n可用工具:\n{tools}\n【历史】\n{history}\nJSON:"

    def _agent(task: "ToolTask", history: list) -> dict:
        prompt = tmpl.format(task=task.prompt, tools=env.specs(), history=_format_history(history))
        try:
            data = call_json(prompt, model=model)
        except Exception:  # noqa: BLE001
            return {"action": "final", "answer": ""}
        if str(data.get("action")) == "call_tool":
            return {"action": "call_tool", "tool": data.get("tool", ""), "args": data.get("args") or {}}
        return {"action": "final", "answer": data.get("answer")}

    return _agent


# --------------------------------------------------------------------------- #
# Episode loop + scoring                                                        #
# --------------------------------------------------------------------------- #


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[\s，,。.；;:：、（）()]", "", s) if s else ""


def run_tool_episode(task: ToolTask, agent: ToolAgent, env: ToolEnv,
                     max_steps: int = 6, model: str = "?") -> ToolEpisodeResult:
    """Drive *agent* over a call→result loop, then score grounding + success."""
    history: list[tuple[str, dict]] = []
    calls: list[ToolCall] = []
    observed: list[ToolResult] = []
    final: Optional[str] = None

    for _ in range(max_steps):
        action = agent(task, history)
        if action.get("action") == "final":
            final = action.get("answer")
            break
        if action.get("action") == "call_tool":
            call = ToolCall(tool=str(action.get("tool", "")), args=dict(action.get("args") or {}))
            calls.append(call)
            res = env.execute(call)
            observed.append(res)
            history.append(("call", call.model_dump()))
            history.append(("result", res.result))
        else:
            break

    called_required = any(c.tool == task.required_tool for c in calls)
    correct = bool(final) and _norm(final) == _norm(task.gold_answer)
    contradiction = False
    for res in observed:
        if res.tool == task.required_tool:
            implied = task.implied_answer(res.result)
            if implied and final and _norm(implied) != _norm(final):
                contradiction = True
    grounded = called_required and not contradiction
    return ToolEpisodeResult(
        task_id=task.task_id, model=model, steps=len(calls), tool_calls=calls,
        final_answer=final, correct=correct, called_required=called_required,
        tool_contradiction=contradiction, grounded=grounded,
        success=correct and grounded,
    )


def aggregate(results: list[ToolEpisodeResult]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "task_success_rate": round(sum(r.success for r in results) / n, 4),
        "tool_use_rate": round(sum(r.called_required for r in results) / n, 4),
        "grounded_rate": round(sum(r.grounded for r in results) / n, 4),
        "contradiction_rate": round(sum(r.tool_contradiction for r in results) / n, 4),
        "answer_accuracy": round(sum(r.correct for r in results) / n, 4),
    }


# --------------------------------------------------------------------------- #
# Demo tasks + orchestration                                                    #
# --------------------------------------------------------------------------- #

_VERDICT = {"true": "有禁忌", "false": "安全"}


def demo_tasks() -> list[ToolTask]:
    return [
        ToolTask(
            task_id="contra-conflict",
            prompt="判断该中药处方是否存在配伍禁忌。方药：半夏、附子、甘草、人参。请给出结论（安全/有禁忌）。",
            required_tool="contraindication_check", gold_answer="有禁忌",
            verdict_key="conflict", verdict_map=_VERDICT,
        ),
        ToolTask(
            task_id="contra-safe",
            prompt="判断该中药处方是否存在配伍禁忌。方药：人参、白术、茯苓、甘草。请给出结论（安全/有禁忌）。",
            required_tool="contraindication_check", gold_answer="安全",
            verdict_key="conflict", verdict_map=_VERDICT,
        ),
    ]


def run(model: str = "mock", cfg: Optional[Config] = None) -> dict:
    """Run the T3 tool-use episodes for *model* over the demo tasks → ``results/``."""
    cfg = cfg or load_config()
    env = ToolEnv()
    agent = llm_tool_agent(model, env, cfg)
    results = [run_tool_episode(t, agent, env, model=model) for t in demo_tasks()]
    metrics = aggregate(results)
    payload = {"model": model, "metrics": metrics, "episodes": [r.model_dump() for r in results]}
    out = ensure_parent(cfg.path("paths.results_dir") / f"tools_{_slug(model)}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("t3[%s] = %s", model, metrics)
    return payload


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


if __name__ == "__main__":
    run()
