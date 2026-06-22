"""ZhongJing-TCM benchmark pipeline — CLI orchestration entry point.

Each sub-command maps to one pipeline stage (M1–M9).  Run a single stage, or
``python run.py pipeline`` to execute M1→M7 end-to-end (works fully offline with
``ZHONGJING_LLM_PROVIDER=mock``).

Examples
--------
    python run.py ingest
    python run.py topics
    python run.py evaluate --model gpt-4o
    ZHONGJING_LLM_PROVIDER=mock python run.py pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer

from config import load_config  # noqa: E402

app = typer.Typer(add_completion=False, help="ZhongJing-TCM benchmark pipeline.")


@app.command()
def ingest() -> None:
    """M1 + M2 — ingest/clean/de-dup, then quality-score & gate articles."""
    import m1_ingest
    import m2_quality

    m1_ingest.run()
    m2_quality.run()


@app.command()
def quality(no_llm: bool = typer.Option(False, help="Skip the LLM judge (heuristic gate only).")) -> None:
    """M2 — quality scoring & gating only."""
    import m2_quality

    m2_quality.run(llm_judge=not no_llm)


@app.command()
def topics() -> None:
    """M3 — passage chunking + BERTopic modelling."""
    import m3_topic

    m3_topic.run()


@app.command()
def label() -> None:
    """M4 — nine-category mapping; emits topic cards & labelled passages."""
    import m4_label

    m4_label.run()


@app.command()
def generate(
    limit: int = typer.Option(0, help="Cap the number of passages (0 = all)."),
    resume: bool = typer.Option(True, help="Resume from existing questions_raw.jsonl."),
    concurrency: int = typer.Option(0, help="Parallel passages in flight (0 = config)."),
    progress: bool = typer.Option(True, help="Show the live tqdm progress bar."),
) -> None:
    """M5 — LLM question generation (single/multiple/short × 3 difficulties).

    Resumable & concurrent: re-run after a disconnect to fill only what's missing.
    Progress is shown live and each passage is stored the moment it completes.
    """
    import m5_generate

    m5_generate.run(
        limit=limit or None, resume=resume, concurrency=concurrency or None, progress=progress
    )


@app.command()
def dtqf() -> None:
    """M6 — Dynamic TCM Question Filtering (core algorithm)."""
    import m6_dtqf

    m6_dtqf.run()


@app.command()
def assemble() -> None:
    """M7 — token-count, split (full + diagnostic), dataset card & figures."""
    import m7_assemble

    m7_assemble.run()


@app.command()
def evaluate(model: str = typer.Option("", help="Model name (default: all in config).")) -> None:
    """M8 — STAGER zero-shot evaluation; updates results/metrics.csv."""
    import m8_evaluate

    if model:
        m8_evaluate.run(model)
    else:
        m8_evaluate.run_all()


@app.command()
def invariance(model: str = typer.Option("", help="Model name (default: all in config).")) -> None:
    """M8 robustness — option-order & label-symbol (A–D↔甲乙丙丁/1–4) invariance."""
    import m8_evaluate

    models = [model] if model else m8_evaluate.load_config().get("evaluate.models", [])
    for m in models:
        m8_evaluate.run_invariance(m)


@app.command()
def consult(
    model: str = typer.Option("mock", help="Expert model under test."),
    max_turns: int = typer.Option(8, help="Max inquiry turns before forcing a diagnosis."),
) -> None:
    """T2 — active-inquiry consultation against the patient simulator (POMDP)."""
    import t2_patient_sim

    t2_patient_sim.run(model=model, max_turns=max_turns)


@app.command()
def counterfactual(
    model: str = typer.Option("", help="Generation model (default: config)."),
    limit: int = typer.Option(0, help="Cap passages (0 = all)."),
) -> None:
    """T1 — generate counterfactual minimal pairs (flip one 四诊 feature → answer flips)."""
    import t1_counterfactual

    t1_counterfactual.run(model=model or t1_counterfactual.load_config().get("generate.model", "gpt-4o"),
                          limit=limit or None)


@app.command()
def process(model: str = typer.Option("mock", help="Model under test / process judge.")) -> None:
    """L2 — step-level process preference (PRM) + result/process gate."""
    import l2_process

    l2_process.run(model=model)


@app.command()
def stats() -> None:
    """M9 — ANOVA + Tukey + regression + DP token segmentation."""
    import m9_stats

    m9_stats.run()


@app.command()
def pipeline(limit: int = typer.Option(0, help="Cap passages for generation (0 = all).")) -> None:
    """Run M1→M7 end-to-end (offline-friendly with the mock LLM provider)."""
    import m1_ingest
    import m2_quality
    import m3_topic
    import m4_label
    import m5_generate
    import m6_dtqf
    import m7_assemble

    load_config()  # validate config early
    m1_ingest.run()
    m2_quality.run()
    m3_topic.run()
    m4_label.run()
    # Fresh full run: regenerate from scratch (passage IDs are new each pipeline).
    # Use `python run.py generate` for the resumable/incremental batch path.
    m5_generate.run(limit=limit or None, resume=False)
    m6_dtqf.run()
    m7_assemble.run()


if __name__ == "__main__":
    app()
