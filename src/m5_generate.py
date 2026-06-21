"""M5 — LLM question generation (human-in-the-loop).

For every labelled passage, generate ``single_choice × multiple_response ×
short_answer`` questions across ``basic / intermediate / advanced`` difficulty
(9 items per passage) with a stepwise explanation and a structured JSON payload.

The generation prompt is **versioned** (``prompts/gen_question.v3.txt``); see
``prompts/CHANGELOG.md`` for the physician-driven refinement history (§5.2).
Cost control (cache / back-off / concurrency) lives in ``llm_client``.
"""
from __future__ import annotations

import itertools
import json
import uuid
from typing import Optional

from config import Config, load_config
from llm_client import call_json
from schemas import DIFFICULTIES, QUESTION_TYPES, Category, Passage, Question
from utils import ensure_parent, get_logger, load_jsonl_as, resolve_path, save_jsonl

_log = get_logger("m5_generate")

QTYPES = list(QUESTION_TYPES)
DIFFS = list(DIFFICULTIES)

_FAILURE_LOG = "data/interim/generation_failures.jsonl"


def log_failure(passage_id: str, qtype: str, diff: str, error: Exception) -> None:
    """Append a generation failure record for later inspection."""
    out = ensure_parent(_FAILURE_LOG)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "passage_id": passage_id,
                    "type": qtype,
                    "difficulty": diff,
                    "error": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _coerce_answer(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in value if str(v).strip()]


def _coerce_question(data: dict, passage: Passage, qtype: str, diff: str) -> Question:
    return Question(
        question_id=str(uuid.uuid4()),
        source_passage_id=passage.passage_id,
        category=passage.category,
        topic_id=int(passage.topic_id),
        type=qtype,
        difficulty=diff,
        stem=str(data["stem"]).strip(),
        options={str(k): str(v) for k, v in (data.get("options") or {}).items()},
        answer=_coerce_answer(data.get("answer")),
        reference_answer=data.get("reference_answer"),
        explanation=str(data.get("explanation", "")).strip(),
        theoretical_basis=data.get("theoretical_basis"),
    )


def generate_for_passage(passage: Passage, model: str = "gpt-4o",
                         max_chars: int = 1500) -> list[Question]:
    """Generate the full 3×3 grid of questions for one labelled passage."""
    if passage.category is None or passage.topic_id is None:
        return []
    cfg = load_config()
    tmpl = resolve_path(cfg.get("prompts.gen_question")).read_text(encoding="utf-8")
    out: list[Question] = []
    for qtype, diff in itertools.product(QTYPES, DIFFS):
        prompt = tmpl.format(qtype=qtype, difficulty=diff, passage_text=passage.text[:max_chars])
        try:
            data = call_json(prompt, model=model)
            out.append(_coerce_question(data, passage, qtype, diff))
        except Exception as exc:  # noqa: BLE001
            log_failure(passage.passage_id, qtype, diff, exc)
    return out


def run(cfg: Optional[Config] = None, limit: Optional[int] = None) -> list[Question]:
    """Generate questions for every labelled passage → ``interim/questions_raw.jsonl``.

    Uses :func:`llm_client.map_async` for bounded-concurrency parallel generation
    (``llm.max_concurrency`` passages in flight simultaneously).
    """
    import asyncio

    from llm_client import map_async

    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    passages = load_jsonl_as(interim / "passages_labeled.jsonl", Passage)
    passages = [p for p in passages if p.category is not None and p.topic_id is not None]
    if limit:
        passages = passages[:limit]

    model = cfg.get("generate.model", "gpt-4o")
    max_chars = cfg.get("generate.max_passage_chars", 1500)
    concurrency = cfg.get("llm.max_concurrency", 4)

    def _gen(p: Passage) -> list[Question]:
        return generate_for_passage(p, model=model, max_chars=max_chars)

    async def _run_all() -> list[Question]:
        results = await map_async(passages, _gen, max_concurrency=concurrency)
        return [q for qs in results for q in qs]

    questions = asyncio.run(_run_all())

    _log.info("generated %d raw questions from %d passages", len(questions), len(passages))
    save_jsonl(questions, interim / "questions_raw.jsonl")
    return questions


if __name__ == "__main__":
    run()
