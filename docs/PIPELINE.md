# ZhongJing-TCM — Pipeline Implementation Guide

End-to-end engineering implementation of the ZhongJing-TCM benchmark: from raw
WeChat-public-account `.txt` articles to a statistically analysable synthetic
benchmark (questions + evaluation + statistics).

> **Input** — a directory of `.txt` files, one cleaned article each
> (`data/raw/账号__标题.txt`).
> **Output** — the 9-category × 3-difficulty × 3-type question bank
> (`data/final/*.jsonl`), plus evaluation metrics and statistical artefacts
> (`results/`).
>
> Because the input is already an article collection, the paper's account-level
> screening is treated as *pre-completed*; M2 reproduces the three-dimension
> expert assessment at the **article** level instead.

## Quickstart

```bash
make install                      # pip install -r requirements.txt
make test                         # 54 unit tests (offline, mock LLM)
ZHONGJING_LLM_PROVIDER=mock make demo   # run M1->M7 fully offline
```

For real generation/evaluation, set credentials and pick a provider:

```bash
export OPENAI_API_KEY=sk-...
# optional OpenAI-compatible endpoint (vLLM / DeepSeek / Together / ...):
export OPENAI_BASE_URL=https://api.your-host/v1
python run.py pipeline            # M1..M7
python run.py evaluate --model gpt-4o
python run.py stats
```

## Pipeline stages

| Stage | Module | CLI | Output |
|---|---|---|---|
| M1 Ingest & clean | `src/m1_ingest.py` | `run.py ingest` | `interim/articles.jsonl` |
| M2 Quality gate | `src/m2_quality.py` | `run.py quality` | `interim/articles_scored.jsonl` |
| M3 Chunk + BERTopic | `src/m3_topic.py` | `run.py topics` | `interim/passages_topiced.jsonl` |
| M4 9-category mapping | `src/m4_label.py` | `run.py label` | `interim/passages_labeled.jsonl`, `topic_cards.csv` |
| M5 Question generation | `src/m5_generate.py` | `run.py generate` | `interim/questions_raw.jsonl` |
| M6 DTQF filtering ★ | `src/m6_dtqf.py` | `run.py dtqf` | `interim/questions_qc.jsonl` |
| M7 Assembly | `src/m7_assemble.py` | `run.py assemble` | `final/*.jsonl`, `dataset_card.csv`, figures |
| M8 Evaluation | `src/m8_evaluate.py` | `run.py evaluate` | `results/metrics.csv`, `results/eval_*.jsonl` |
| M9 Statistics ★ | `src/m9_stats.py` | `run.py stats` | `results/anova.csv`, `segments.json`, figures |

Shared infrastructure: `schemas.py` (pydantic records), `llm_client.py`
(cache + retry + providers, incl. an offline **mock**), `config.py`
(`configs/pipeline.yaml` loader), `utils.py` (JSONL I/O, paths, logging).

## Paper formula ↔ code index

| Paper | Meaning | Location |
|---|---|---|
| Eq. 1 | sentence-BERT embeddings | `m3_topic.build_topic_model` |
| Eq. 2 | HDBSCAN clustering | `m3_topic.build_topic_model` |
| Eq. 3 | c-TF-IDF topic representation | `m3_topic` `CountVectorizer(tokenizer=jieba_tok)` |
| Eq. 4 | labelling function `f` | `m4_label.suggest_category` (+ physician review) |
| Eq. 5 | initial stratified sampling (S=100) | `m6_dtqf.dtqf_filter` (`it == 0`) |
| Eq. 6 | error-rate-driven sampling | `m6_dtqf.dtqf_filter` (`else`) |
| Eq. 7 | TF-IDF(t, q) | `m6_dtqf.KeywordExtractor.extract` |
| Eq. 8 | word↔question cosine | `m6_dtqf.KeywordExtractor.extract` |
| Eq. 9 | Cohen's kappa ≥ 0.8 | `m4_label.review_agreement` |
| Termination | `P_qualified > 95%` | `m6_dtqf.dtqf_filter` (probe) |
| Algorithm 1 | DP token segmentation | `m9_stats.optimal_segmentation` |

## Design notes

- **Offline-first.** `ZHONGJING_LLM_PROVIDER=mock` makes every LLM call
  deterministic and free, so the whole pipeline and the test-suite run with no
  API key or network. The mock returns schema-valid questions / quality scores /
  STAGER answers keyed off the prompt family.
- **Lazy heavy deps.** `sentence-transformers`, `bertopic`, `umap`, `hdbscan`,
  `gensim`, `matplotlib`, `openai`, `opencc`, `tiktoken` are imported inside the
  functions that need them, so the testable core stays importable in a minimal
  environment. `tiktoken` falls back to a CJK-aware heuristic when its encoding
  file can't be downloaded.
- **Human-in-the-loop.** M4 emits `topic_cards.csv` for physician review (drop a
  completed `topic_cards_reviewed.csv` back to override the auto labels). M5's
  generation prompt is versioned (`prompts/CHANGELOG.md`).
- **Leakage defence.** M1 de-duplicates near-identical reposts (MinHash/LSH);
  only **synthetic** questions are released — never the source articles
  (copyright + privacy, manual §10.4).

## Two corrections vs. the source manual

While implementing, two latent bugs in the manual's sketch code were fixed:

1. `prompts/judge_quality.txt` — the JSON braces are escaped (`{{ }}`) so the
   `str.format(article_text=...)` call doesn't raise `KeyError`.
2. `optimal_segmentation` (Algorithm 1) back-tracking returns `points[1:]`
   (the internal break-points) rather than `points[:-1]`, which had discarded
   the actual break-points and returned the right edge instead. Verified by
   `tests/test_segmentation.py`.
