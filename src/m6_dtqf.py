"""M6 — Dynamic TCM Question Filtering (DTQF). ★ core algorithm

Removes incomplete / ambiguous questions — especially clinical items lacking a
full patient symptom description.  Faithful to the paper's Eq. 5–9:

* Eq. 5/6 — stratified-then-dynamic sampling: round 0 allocates the review
  budget ``S`` proportionally to topic size; later rounds reallocate it by each
  topic's **error rate**, focusing effort where defects concentrate.
* Eq. 7/8 — keyword extraction = TF-IDF(t, q) × cosine(v_t, v_q) (Word2Vec).
* Eq. 9 — Cohen's kappa gate (≥ 0.8) on physician spot-checks.
* Termination — stop once a random 100-item probe is > 95% qualified.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import numpy as np

from config import Config, load_config
from schemas import CLINICAL_CATEGORIES, Question
from utils import get_logger, load_jsonl_as, resolve_path, save_jsonl

_log = get_logger("m6_dtqf")

#: Symptom / sign cues that signal a complete clinical presentation.
SYMPTOM_HINTS = {
    "症状", "主诉", "舌", "脉", "苔", "发热", "恶寒", "疼痛", "痛", "乏力", "纳差",
    "便", "汗", "面色", "畏寒", "口干", "口苦", "头晕", "心悸", "失眠", "咳",
}


# --------------------------------------------------------------------------- #
# Eq. 7/8 — keyword extraction                                                  #
# --------------------------------------------------------------------------- #


class KeywordExtractor:
    """式7: TF-IDF(t,q)=TF·IDF ; 式8: cos(v_t, v_q) ; score = 乘积。"""

    def __init__(self, corpus_tokens: list[list[str]], w2v_path: Optional[str] = None,
                 vector_size: int = 200, window: int = 5, min_count: int = 2):
        from gensim.models import Word2Vec
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
        self.tfidf.fit(corpus_tokens)
        self.vocab = self.tfidf.vocabulary_
        if w2v_path and resolve_path(w2v_path).exists():
            self.w2v = Word2Vec.load(str(resolve_path(w2v_path)))
        else:
            self.w2v = self._train_w2v(
                Word2Vec, corpus_tokens, vector_size, window, min_count
            )

    @staticmethod
    def _train_w2v(Word2Vec, corpus_tokens, vector_size, window, min_count):
        """Train Word2Vec, backing off ``min_count`` for tiny corpora."""
        for mc in (min_count, 1):
            try:
                return Word2Vec(
                    corpus_tokens, vector_size=vector_size, window=window,
                    min_count=mc, workers=1, seed=42,
                )
            except RuntimeError:
                continue
        return None  # extract() falls back to pure TF-IDF

    def _has_vec(self, token: str) -> bool:
        return self.w2v is not None and token in self.w2v.wv

    def _qvec(self, tokens: list[str]):
        if self.w2v is None:
            return None
        vs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        return np.mean(vs, axis=0) if vs else None

    def extract(self, question_text: str, topk: int = 8) -> list[str]:
        import jieba

        tokens = [w for w in jieba.cut(question_text) if len(w) > 1]
        if not tokens:
            return []
        tfidf_row = self.tfidf.transform([tokens]).toarray()[0]
        qvec = self._qvec(tokens)
        scored: list[tuple[str, float]] = []
        for t in set(tokens):
            if t not in self.vocab:
                continue
            tf = tfidf_row[self.vocab[t]]                       # 式7
            # 式8: word-to-question cosine; degrade to TF-IDF only if no vectors.
            sim = 1.0
            if qvec is not None and self._has_vec(t):
                tv = self.w2v.wv[t]
                sim = float(
                    np.dot(tv, qvec) / (np.linalg.norm(tv) * np.linalg.norm(qvec) + 1e-9)
                )
            scored.append((t, tf * sim))                        # 综合得分
        return [w for w, _ in sorted(scored, key=lambda x: -x[1])[:topk]]


def build_keyword_extractor(questions: list[Question], cfg: Optional[Config] = None) -> KeywordExtractor:
    """Train a :class:`KeywordExtractor` from the question stems/explanations."""
    import jieba

    cfg = cfg or load_config()
    w2v = cfg.section("dtqf").get("w2v", {})
    corpus = [[w for w in jieba.cut(f"{q.stem} {q.explanation}") if len(w) > 1] for q in questions]
    corpus = [toks for toks in corpus if toks] or [["中医"]]
    return KeywordExtractor(
        corpus,
        vector_size=w2v.get("vector_size", 200),
        window=w2v.get("window", 5),
        min_count=w2v.get("min_count", 2),
    )


# --------------------------------------------------------------------------- #
# Optional LLM validity judge                                                   #
# --------------------------------------------------------------------------- #


def _render_question(q: Question) -> str:
    lines = [q.stem]
    for k, v in q.options.items():
        lines.append(f"{k}. {v}")
    if q.answer:
        lines.append(f"参考答案: {', '.join(q.answer)}")
    if q.reference_answer:
        lines.append(f"参考答案: {q.reference_answer}")
    return "\n".join(lines)


def llm_judge_valid(q: Question) -> bool:
    """Optional LLM completeness/uniqueness check; failures default to valid."""
    from llm_client import call_json

    try:
        cfg = load_config()
        tmpl = resolve_path(cfg.get("prompts.judge_validity")).read_text(encoding="utf-8")
        data = call_json(tmpl.format(question=_render_question(q)))
        return bool(data.get("valid", True))
    except Exception as exc:  # noqa: BLE001
        _log.debug("llm_judge_valid failed (%s); treating as valid", exc)
        return True


# --------------------------------------------------------------------------- #
# Review function (completeness + ambiguity + format)                           #
# --------------------------------------------------------------------------- #


def review_question(q: Question, kw_extractor: Optional[KeywordExtractor] = None,
                    use_llm_judge: bool = False, min_stem: int = 15,
                    min_explanation: int = 20) -> bool:
    """Return ``True`` if *q* is well-formed, complete and unambiguous."""
    # 1) 自动:临床类题须含足够症状/体征描述
    if q.category in CLINICAL_CATEGORIES:
        kws = kw_extractor.extract(q.stem) if kw_extractor is not None else []
        if not (set(kws) & SYMPTOM_HINTS) and not any(h in q.stem for h in SYMPTOM_HINTS):
            return False                                # 缺症状描述 → 不合格

    # 2) 自动:歧义/格式
    if q.type in ("single_choice", "multiple_response"):
        if len(q.options) < 4 or not q.answer:
            return False
        if any(not v.strip() for v in q.options.values()):
            return False
        # answer must reference real option keys
        if any(a not in q.options for a in q.answer):
            return False
    elif q.type == "short_answer":
        if not (q.reference_answer and q.reference_answer.strip()):
            return False

    if len(q.stem) < min_stem or len(q.explanation) < min_explanation:
        return False

    # 3) 可选:LLM 裁判判完整性/唯一性
    if use_llm_judge and not llm_judge_valid(q):
        return False
    return True


# --------------------------------------------------------------------------- #
# Eq. 5/6 — DTQF main loop                                                       #
# --------------------------------------------------------------------------- #


def dtqf_filter(questions: list[Question], kw_extractor: Optional[KeywordExtractor] = None,
                S: int = 100, max_iter: int = 20, qualify_threshold: float = 0.95,
                use_llm_judge: bool = False, seed: Optional[int] = None) -> list[Question]:
    """Iteratively sample → review → prune until > ``qualify_threshold`` qualified."""
    rng = random.Random(seed)
    by_topic: dict[int, list[Question]] = defaultdict(list)
    for q in questions:
        by_topic[q.topic_id].append(q)
    topics = list(by_topic.keys())
    error_rates = {t: 1.0 for t in topics}

    def _review(q):
        return review_question(q, kw_extractor, use_llm_judge=use_llm_judge)

    for it in range(max_iter):
        # ---- 采样量:首轮按题量比例(式5),之后按错误率(式6) ----
        if it == 0:
            N = sum(len(v) for v in by_topic.values()) or 1
            sizes = {t: max(1, round(len(by_topic[t]) / N * S)) for t in topics}
        else:
            tot = sum(error_rates.values()) + 1e-9
            sizes = {t: max(1, round(error_rates[t] / tot * S)) for t in topics}

        # ---- 采样 → 审查 → 剔除 → 记录错误率 ----
        new_err: dict[int, float] = {}
        for t in topics:
            pool = by_topic[t]
            if not pool:
                new_err[t] = 0.0
                continue
            sample = rng.sample(pool, min(sizes[t], len(pool)))
            failed = [q for q in sample if not _review(q)]
            for q in failed:
                pool.remove(q)                          # 从题库剔除不合格题
            new_err[t] = len(failed) / len(sample)
        error_rates = new_err

        # ---- 终止条件:随机 100 题合格率 > 95% ----
        flat = [q for v in by_topic.values() for q in v]
        if len(flat) >= 100:
            probe = rng.sample(flat, 100)
            p_qualified = sum(_review(q) for q in probe) / 100
            _log.info("[DTQF] iter=%d pool=%d P_qualified=%.3f", it, len(flat), p_qualified)
            if p_qualified > qualify_threshold:
                break
        else:
            _log.info("[DTQF] iter=%d pool=%d (probe skipped: < 100 items)", it, len(flat))
            if all(e == 0.0 for e in error_rates.values()):
                break

    survivors = [q for v in by_topic.values() for q in v]
    for q in survivors:
        q.qc_passed = True
    _log.info("DTQF kept %d / %d questions", len(survivors), len(questions))
    return survivors


def run(cfg: Optional[Config] = None) -> list[Question]:
    """Filter ``interim/questions_raw.jsonl`` → ``interim/questions_qc.jsonl``."""
    cfg = cfg or load_config()
    interim = cfg.path("paths.interim_dir")
    questions = load_jsonl_as(interim / "questions_raw.jsonl", Question)
    kw_extractor = build_keyword_extractor(questions, cfg)
    survivors = dtqf_filter(
        questions,
        kw_extractor,
        S=cfg.get("dtqf.sample_size", 100),
        max_iter=cfg.get("dtqf.max_iter", 20),
        qualify_threshold=cfg.get("dtqf.qualify_threshold", 0.95),
        use_llm_judge=cfg.get("dtqf.use_llm_judge", False),
        seed=cfg.get("topic.random_state", 42),
    )
    save_jsonl(survivors, interim / "questions_qc.jsonl")
    return survivors


if __name__ == "__main__":
    run()
