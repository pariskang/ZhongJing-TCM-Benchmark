"""Tests for the LLM client: JSON extraction, mock provider, caching."""
import pytest

from llm_client import LLMClient, extract_json, mock_completion


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = "好的，结果如下:\n```json\n{\"answer\": [\"A\"]}\n```\n谢谢"
    assert extract_json(text) == {"answer": ["A"]}


def test_extract_json_with_prose():
    text = '分析后给出 {"professionalism": 8, "popularization": 7} 这样的结果'
    out = extract_json(text)
    assert out["professionalism"] == 8


def test_extract_json_trailing_comma_via_literal():
    # ast.literal_eval already tolerates a trailing comma (no json-repair needed).
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_extract_json_repair_truncated():
    # Truncated/missing closing brace: only json-repair can recover it.
    pytest.importorskip("json_repair")
    out = extract_json('{"stem": "证属肝郁", "answer": ["A"]')  # no closing }
    assert out["answer"] == ["A"]


def test_extract_json_repair_unquoted_keys():
    pytest.importorskip("json_repair")
    out = extract_json("前缀 {stem: '气虚', answer: ['B']} 后缀")
    assert out["answer"] == ["B"]


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("完全没有 JSON 的文本")


@pytest.mark.parametrize("qtype,has_opts,has_ref", [
    ("single_choice", True, False),
    ("multiple_response", True, False),
    ("short_answer", False, True),
])
def test_mock_generation_per_type(qtype, has_opts, has_ref):
    prompt = f"题型: {qtype}\n源文本\n\"\"\"测试文本\"\"\""
    data = extract_json(mock_completion(prompt))
    assert bool(data["options"]) is has_opts
    assert (data.get("reference_answer") is not None) is has_ref
    assert len(data["explanation"]) > 10


def test_mock_quality_judge():
    data = extract_json(mock_completion("专业性 科普性 实用性 professionalism popularization"))
    for k in ("professionalism", "popularization", "practicality"):
        assert 0 <= data[k] <= 10


def test_cache_hit(tmp_path):
    client = LLMClient(provider="mock", cache_dir=str(tmp_path / "c"))
    p = "题型: single_choice\n源文本"
    first = client.call_json(p, model="mock")
    second = client.call_json(p, model="mock")
    assert first == second


def test_mock_overrides_provider():
    # Even with an openai-configured client, model='mock' routes to the mock.
    client = LLMClient(provider="openai")
    assert client._resolve_provider("mock-x") == "mock"
    assert client.call_text("题型: single_choice\n源文本", model="mock").strip()
