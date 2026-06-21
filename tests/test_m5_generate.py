"""Tests for M5 question generation (mock provider)."""
import asyncio
import uuid

from m5_generate import generate_for_passage
from schemas import Category, Passage


def _passage():
    return Passage(
        passage_id=str(uuid.uuid4()), article_id="a",
        text="小柴胡汤和解少阳，主治往来寒热、胸胁苦满、口苦咽干、脉弦。柴胡为君药。",
        topic_id=0, category=Category.CLASSIC_FORMULA, topic_keywords=["柴胡", "少阳"],
    )


def test_generates_full_grid():
    qs = generate_for_passage(_passage(), model="mock")
    assert len(qs) == 9  # 3 types x 3 difficulties
    assert {q.type for q in qs} == {"single_choice", "multiple_response", "short_answer"}
    assert {q.difficulty for q in qs} == {"basic", "intermediate", "advanced"}


def test_payload_matches_type():
    qs = {q.type: q for q in generate_for_passage(_passage(), model="mock")}
    assert qs["single_choice"].answer and qs["single_choice"].options
    assert qs["multiple_response"].options
    assert qs["short_answer"].options == {}
    assert qs["short_answer"].reference_answer


def test_skips_unlabelled_passage():
    p = _passage()
    p.category = None
    assert generate_for_passage(p, model="mock") == []


def test_async_generation_two_passages():
    from llm_client import map_async

    async def _run():
        results = await map_async([_passage(), _passage()], generate_for_passage, max_concurrency=2)
        return results

    results = asyncio.run(_run())
    assert len(results) == 2
    assert all(len(qs) == 9 for qs in results)
