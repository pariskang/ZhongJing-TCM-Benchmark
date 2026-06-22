"""Tests for T3 tool-use agent: checkers, grounding, and contradiction detection."""
from t3_tools import (
    ToolEnv,
    aggregate,
    contraindication_check,
    demo_tasks,
    dose_check,
    llm_tool_agent,
    run_tool_episode,
    scripted_tool_agent,
)


# -- the deterministic checkers ---------------------------------------------- #


def test_contraindication_check_detects_shibanfan():
    out = contraindication_check(["半夏", "附子", "甘草", "人参"])
    assert out["conflict"] is True
    assert ["半夏", "附子"] in out["pairs"] or ["附子", "半夏"] in out["pairs"]


def test_contraindication_check_safe():
    assert contraindication_check(["人参", "白术", "茯苓", "甘草"])["conflict"] is False


def test_contraindication_check_shijiuwei():
    assert contraindication_check(["人参", "五灵脂"])["conflict"] is True


def test_dose_check_overdose():
    assert dose_check("附子", 30)["over"] is True
    assert dose_check("附子", 10)["over"] is False


# -- episode loop + grounding ------------------------------------------------ #


def _conflict_task():
    return demo_tasks()[0]   # 半夏+附子 → 有禁忌


def test_mock_agent_grounds_answer_in_tool():
    env = ToolEnv()
    res = run_tool_episode(_conflict_task(), llm_tool_agent("mock", env), env)
    assert res.called_required is True
    assert res.final_answer == "有禁忌"
    assert res.correct is True
    assert res.tool_contradiction is False
    assert res.grounded is True and res.success is True


def test_answer_without_tool_is_ungrounded():
    env = ToolEnv()
    agent = scripted_tool_agent([{"action": "final", "answer": "安全"}])
    res = run_tool_episode(_conflict_task(), agent, env)
    assert res.called_required is False
    assert res.grounded is False
    assert res.success is False


def test_contradicting_tool_result_is_flagged():
    # Calls the checker (sees a conflict) but still answers "安全" → contradiction.
    env = ToolEnv()
    agent = scripted_tool_agent([
        {"action": "call_tool", "tool": "contraindication_check", "args": {"herbs": ["半夏", "附子"]}},
        {"action": "final", "answer": "安全"},
    ])
    res = run_tool_episode(_conflict_task(), agent, env)
    assert res.called_required is True
    assert res.tool_contradiction is True
    assert res.grounded is False
    assert res.success is False


def test_aggregate_over_demo_tasks_with_mock():
    env = ToolEnv()
    agent = llm_tool_agent("mock", env)
    results = [run_tool_episode(t, agent, env) for t in demo_tasks()]
    m = aggregate(results)
    assert m["n"] == 2
    assert m["task_success_rate"] == 1.0     # mock calls the tool and grounds both
    assert m["tool_use_rate"] == 1.0
    assert m["contradiction_rate"] == 0.0
