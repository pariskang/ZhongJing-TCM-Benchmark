"""M5 — LLM question generation (human-in-the-loop, resumable, concurrent).

For every labelled passage, generate ``single_choice × multiple_response ×
short_answer`` questions across ``basic / intermediate / advanced`` difficulty
(9 items per passage) with a stepwise explanation and a structured JSON payload.

The generation prompt is **versioned** (``prompts/gen_question.v3.txt``); see
``prompts/CHANGELOG.md`` for the physician-driven refinement history (§5.2).
Cost control (cache / back-off / concurrency) lives in ``llm_client``.

Batch features
--------------
* **Concurrency.** Passages are generated in parallel with a bounded
  ``asyncio.Semaphore`` (``llm.max_concurrency``), so MiniMax / OpenAI calls
  overlap instead of running one-at-a-time.
* **Checkpoint / resume.** Each passage's questions are appended to
  ``questions_raw.jsonl`` as soon as they finish.  Re-running with
  ``resume=True`` (the default) reads which ``(passage, type, difficulty)``
  triples already exist and regenerates only what's missing — a Colab runtime
  disconnect costs nothing.
"""
from __future__ import annotations

import itertools
import json
import uuid
from typing import Optional

from config import Config, load_config
from llm_client import call_json
from schemas import DIFFICULTIES, QUESTION_TYPES, Category, Passage, Question
from utils import (
    append_jsonl,
    ensure_parent,
    get_logger,
    iter_jsonl,
    load_jsonl_as,
    resolve_path,
)

_log = get_logger("m5_generate")

QTYPES = list(QUESTION_TYPES)
DIFFS = list(DIFFICULTIES)
FULL_GRID: list[tuple[str, str]] = list(itertools.product(QTYPES, DIFFS))

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


def generate_for_passage(
    passage: Passage,
    model: str = "gpt-4o",
    max_chars: int = 1500,
    combos: Optional[list[tuple[str, str]]] = None,
) -> list[Question]:
    """Generate questions for one labelled passage.

    *combos* selects which ``(type, difficulty)`` pairs to produce; when ``None``
    the full 3×3 grid is generated.  Resume passes only the missing pairs.
    """
    if passage.category is None or passage.topic_id is None:
        return []
    cfg = load_config()
    tmpl = resolve_path(cfg.get("prompts.gen_question")).read_text(encoding="utf-8")
    language = cfg.get("generate.language", "简体中文")
    out: list[Question] = []
    for qtype, diff in (combos if combos is not None else FULL_GRID):
        prompt = tmpl.format(
            qtype=qtype, difficulty=diff, language=language,
            passage_text=passage.text[:max_chars],
        )
        try:
            data = call_json(prompt, model=model)
            out.append(_coerce_question(data, passage, qtype, diff))
        except Exception as exc:  # noqa: BLE001
            log_failure(passage.passage_id, qtype, diff, exc)
    return out


def _done_combos(path) -> set[tuple[str, str, str]]:
    """Read which ``(passage_id, type, difficulty)`` triples already exist."""
    done: set[tuple[str, str, str]] = set()
    p = resolve_path(path)
    if not p.exists():
        return done
    for rec in iter_jsonl(p):
        done.add((rec.get("source_passage_id"), rec.get("type"), rec.get("difficulty")))
    return done


def _make_bar(total: int, desc: str = "M5 生成试题"):
    """A live tqdm progress bar (notebook-aware), or ``None`` if tqdm is absent."""
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, unit="passage")
    except Exception:  # pragma: no cover - optional dep
        return None


async def _generate_all(work, model, max_chars, concurrency, out_path, progress: bool = True) -> int:
    """Generate *work* (passage, missing-combos) pairs concurrently, checkpointing.

    Progress is reported live: each passage advances a tqdm bar (or, without
    tqdm, logs every 25 passages), and its questions are flushed to *out_path*
    the moment the passage completes — so storage is real-time, not batched.
    """
    import asyncio

    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    total_new = 0
    done_passages = 0
    total = len(work)
    bar = _make_bar(total) if progress else None

    async def worker(passage: Passage, combos: list[tuple[str, str]]):
        nonlocal total_new, done_passages
        async with sem:
            qs = await asyncio.to_thread(generate_for_passage, passage, model, max_chars, combos)
        async with write_lock:                       # checkpoint as each passage lands
            if qs:
                append_jsonl(qs, out_path)
                total_new += len(qs)
            done_passages += 1
            if bar is not None:
                bar.update(1)
                bar.set_postfix(questions=total_new)
            elif done_passages % 25 == 0 or done_passages == total:
                _log.info("progress: %d/%d passages (+%d questions)", done_passages, total, total_new)

    try:
        await asyncio.gather(*[worker(p, c) for p, c in work])
    finally:
        if bar is not None:
            bar.close()
    return total_new


def run(
    cfg: Optional[Config] = None,
    limit: Optional[int] = None,
    resume: bool = True,
    concurrency: Optional[int] = None,
    progress: bool = True,
) -> list[Question]:
    """Generate questions for every labelled passage → ``interim/questions_raw.jsonl``.

    Parallel (``llm.max_concurrency``) and resumable: with ``resume=True`` only
    the missing ``(passage, type, difficulty)`` triples are generated and each
    passage is checkpointed to disk the moment it completes.  Set ``progress``
    to show/hide the live tqdm bar.
    """
    import asyncio

    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    out_path = ensure_parent(interim / "questions_raw.jsonl")

    passages = load_jsonl_as(interim / "passages_labeled.jsonl", Passage)
    passages = [p for p in passages if p.category is not None and p.topic_id is not None]
    if limit:
        passages = passages[:limit]

    model = cfg.get("generate.model", "gpt-4o")
    max_chars = cfg.get("generate.max_passage_chars", 1500)
    concurrency = concurrency or cfg.get("llm.max_concurrency", 4)

    if resume and out_path.exists():
        done = _done_combos(out_path)
        _log.info("resume: %d (passage,type,difficulty) triples already on disk", len(done))
    else:
        done = set()
        if out_path.exists():
            out_path.unlink()                        # fresh start

    work = []
    for p in passages:
        missing = [(t, d) for (t, d) in FULL_GRID if (p.passage_id, t, d) not in done]
        if missing:
            work.append((p, missing))

    if not work:
        _log.info("nothing to generate — all %d passages complete", len(passages))
        return load_jsonl_as(out_path, Question) if out_path.exists() else []

    _log.info(
        "generating %d passages (%d triples) with concurrency=%d, model=%s",
        len(work), sum(len(c) for _, c in work), concurrency, model,
    )
    new_count = asyncio.run(
        _generate_all(work, model, max_chars, concurrency, out_path, progress=progress)
    )

    questions = load_jsonl_as(out_path, Question)
    _log.info("generated %d new questions; %d total in %s", new_count, len(questions), out_path.name)
    return questions


if __name__ == "__main__":
    run()
