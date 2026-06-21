"""Tests for M5 question generation (mock provider)."""
import asyncio
import uuid

from config import Config
from m5_generate import FULL_GRID, _done_combos, generate_for_passage
from schemas import Category, Passage
from utils import save_jsonl


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


def test_generate_subset_of_combos():
    combos = [("single_choice", "basic"), ("short_answer", "advanced")]
    qs = generate_for_passage(_passage(), model="mock", combos=combos)
    assert len(qs) == 2
    assert {(q.type, q.difficulty) for q in qs} == set(combos)


def _cfg(tmp_path, concurrency=4):
    return Config(
        {
            "paths": {"interim_dir": str(tmp_path)},
            "generate": {"model": "mock", "max_passage_chars": 1500},
            "llm": {"max_concurrency": concurrency},
        }
    )


def test_run_writes_full_grid(tmp_path):
    import m5_generate

    save_jsonl([_passage(), _passage()], tmp_path / "passages_labeled.jsonl")
    qs = m5_generate.run(cfg=_cfg(tmp_path), resume=False)
    assert len(qs) == 18  # 2 passages × 9
    assert (tmp_path / "questions_raw.jsonl").exists()


def test_run_resume_skips_completed(tmp_path):
    import m5_generate

    save_jsonl([_passage(), _passage()], tmp_path / "passages_labeled.jsonl")
    cfg = _cfg(tmp_path)
    m5_generate.run(cfg=cfg, resume=False)
    out = tmp_path / "questions_raw.jsonl"

    # Second run with resume=True → nothing new, count unchanged.
    qs2 = m5_generate.run(cfg=cfg, resume=True)
    assert len(qs2) == 18


def test_run_resume_regenerates_missing(tmp_path):
    import m5_generate

    save_jsonl([_passage(), _passage()], tmp_path / "passages_labeled.jsonl")
    cfg = _cfg(tmp_path)
    m5_generate.run(cfg=cfg, resume=False)
    out = tmp_path / "questions_raw.jsonl"

    # Simulate a crash: keep only the first 10 of 18 lines.
    lines = out.read_text(encoding="utf-8").splitlines()
    out.write_text("\n".join(lines[:10]) + "\n", encoding="utf-8")
    assert len(_done_combos(out)) <= 10

    qs3 = m5_generate.run(cfg=cfg, resume=True)
    assert len(qs3) == 18  # the missing triples were regenerated
    # every (passage, type, difficulty) triple present exactly once
    triples = _done_combos(out)
    assert len(triples) == 18
