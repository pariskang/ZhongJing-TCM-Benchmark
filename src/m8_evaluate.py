"""M8 — Model evaluation.

Zero-shot evaluation with the STAGER structured prompt (paper Table
``tcm_prompt``).  Each choice question is run ``n_runs`` times and reduced by
majority vote.  Reports Accuracy / Precision / Recall / F1 plus the **refusal
rate** — both explicit refusals ("信息不足" …) and implicit ones (no option
selected).  Per-question records (with output-token lengths) feed M9.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from config import Config, load_config
from llm_client import call_text
from m7_assemble import _encode_len
from schemas import Category, EvalRecord, Question
from utils import get_logger, load_jsonl_as, resolve_path, save_jsonl

_log = get_logger("m8_evaluate")

REFUSAL_PAT = re.compile(
    r"信息不足|无法确定|需要更多|资料不全|不能确定|无法回答|"
    r"cannot determine|insufficient information|not enough information",
    re.IGNORECASE,
)


class ModelEvaluator:
    """Evaluate a single model over a dataset of questions (choice + short-answer)."""

    def __init__(self, model_name: str, cfg: Optional[Config] = None):
        self.model = model_name
        self.cfg = cfg or load_config()
        self.n_runs = self.cfg.get("evaluate.n_runs", 3)
        self.tmpl = resolve_path(self.cfg.get("prompts.stager_eval")).read_text(encoding="utf-8")
        self._sa_tmpl: Optional[str] = self._load_sa_tmpl()

    def _load_sa_tmpl(self) -> Optional[str]:
        try:
            return resolve_path(self.cfg.get("prompts.judge_short_answer")).read_text(encoding="utf-8")
        except Exception:
            return None

    # -- prompt / parse (choice) --------------------------------------------- #
    def build_prompt(self, q: Question) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in q.options.items())
        return self.tmpl.format(question=f"{q.stem}\n{opts}")

    def parse(self, raw: str) -> dict:
        if not raw or REFUSAL_PAT.search(raw):
            return {"refused": True, "pred": []}
        m = re.search(r"\[Answer\]\s*([A-D，,、\s]+)", raw)
        pred = re.findall(r"[A-D]", m.group(1)) if m else []
        return {"refused": not pred, "pred": sorted(set(pred))}

    # -- single choice question ---------------------------------------------- #
    def eval_one(self, q: Question) -> EvalRecord:
        raws = [call_text(self.build_prompt(q), self.model) for _ in range(self.n_runs)]
        runs = [self.parse(r) for r in raws]
        keys = [tuple(r["pred"]) for r in runs if not r["refused"]]
        refused = sum(r["refused"] for r in runs) > self.n_runs // 2
        pred = list(Counter(keys).most_common(1)[0][0]) if keys else []
        gold = sorted(q.answer)
        correct = (not refused) and (pred == gold)
        out_tokens = [_encode_len(r) for r in raws]
        return EvalRecord(
            question_id=q.question_id,
            model=self.model,
            category=q.category,
            difficulty=q.difficulty,
            type=q.type,
            gold=gold,
            pred=pred,
            refused=refused,
            correct=correct,
            answer_tokens=round(sum(out_tokens) / len(out_tokens)) if out_tokens else 0,
            output_tokens=max(out_tokens) if out_tokens else 0,
            raw_output=raws[0] if raws else None,
        )

    # -- short-answer question ----------------------------------------------- #
    def eval_one_short_answer(self, q: Question) -> Optional[EvalRecord]:
        """LLM semantic matching for short-answer questions.

        Generates the model's open-ended response, then uses a judge prompt to
        compare it against ``q.reference_answer``.  Returns ``None`` when no
        judge prompt is configured.
        """
        if self._sa_tmpl is None or not q.reference_answer:
            return None
        raws = [call_text(q.stem + "\n请简要作答。", self.model) for _ in range(self.n_runs)]
        judge_model = self.cfg.get("evaluate.judge_model", self.model)
        correct_votes: list[bool] = []
        for raw in raws:
            judge_prompt = self._sa_tmpl.format(
                question=q.stem,
                reference_answer=q.reference_answer,
                student_answer=raw,
            )
            try:
                from llm_client import call_json as _cj

                result = _cj(judge_prompt, model=judge_model)
                correct_votes.append(bool(result.get("correct", False)))
            except Exception:
                correct_votes.append(False)
        correct = sum(correct_votes) > len(correct_votes) // 2
        out_tokens = [_encode_len(r) for r in raws]
        return EvalRecord(
            question_id=q.question_id,
            model=self.model,
            category=q.category,
            difficulty=q.difficulty,
            type=q.type,
            gold=[q.reference_answer],
            pred=[raws[0]] if raws else [],
            refused=False,
            correct=correct,
            answer_tokens=round(sum(out_tokens) / len(out_tokens)) if out_tokens else 0,
            output_tokens=max(out_tokens) if out_tokens else 0,
            raw_output=raws[0] if raws else None,
        )

    # -- whole dataset ------------------------------------------------------- #
    def evaluate(self, dataset: list[Question]) -> tuple[dict, list[EvalRecord]]:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        rows: list[EvalRecord] = []
        for q in dataset:
            if q.type == "short_answer":
                rec = self.eval_one_short_answer(q)
                if rec is not None:
                    rows.append(rec)
            else:
                rows.append(self.eval_one(q))

        if not rows:
            return {"model": self.model, "n": 0}, rows

        # Choice-question Acc/P/R/F1 (exclude short_answer from sklearn metrics).
        choice_rows = [r for r in rows if r.type != "short_answer"]
        scored = [r for r in choice_rows if not r.refused]
        if scored:
            y_true = [",".join(r.gold) for r in scored]
            y_pred = [",".join(r.pred) for r in scored]
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
        else:
            acc = p = r = f1 = 0.0

        sa_rows = [r for r in rows if r.type == "short_answer"]
        sa_acc = sum(r.correct for r in sa_rows) / len(sa_rows) if sa_rows else None
        refusal_rate = (
            sum(x.refused for x in choice_rows) / len(choice_rows) if choice_rows else 0.0
        )
        metrics: dict = dict(
            model=self.model,
            n=len(rows),
            accuracy=round(acc, 4),
            precision=round(p, 4),
            recall=round(r, 4),
            f1=round(f1, 4),
            refusal_rate=round(refusal_rate, 4),
        )
        if sa_acc is not None:
            metrics["short_answer_accuracy"] = round(sa_acc, 4)
        return metrics, rows


def _load_dataset(cfg: Config) -> list[Question]:
    final = cfg.path("paths.final_dir")
    diagnostic = final / "zhongjing_tcm_diagnostic.jsonl"
    full = final / "zhongjing_tcm_full.jsonl"
    path = diagnostic if diagnostic.exists() else full
    return load_jsonl_as(path, Question)


def run(model: str, cfg: Optional[Config] = None) -> dict:
    """Evaluate one model; write per-question records and update ``metrics.csv``."""
    cfg = cfg or load_config()
    dataset = _load_dataset(cfg)
    metrics, rows = ModelEvaluator(model, cfg).evaluate(dataset)

    results = cfg.path("paths.results_dir")
    save_jsonl(rows, results / f"eval_{_slug(model)}.jsonl")
    _update_metrics_csv(metrics, results / "metrics.csv")
    _log.info("metrics[%s] = %s", model, metrics)
    return metrics


def run_all(cfg: Optional[Config] = None) -> list[dict]:
    cfg = cfg or load_config()
    return [run(m, cfg) for m in cfg.get("evaluate.models", [])]


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _update_metrics_csv(metrics: dict, path) -> None:
    import pandas as pd

    from utils import ensure_parent

    out = ensure_parent(path)
    if out.exists():
        df = pd.read_csv(out)
        df = df[df["model"] != metrics["model"]]
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
    df.sort_values("model").to_csv(out, index=False)


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "mock")
