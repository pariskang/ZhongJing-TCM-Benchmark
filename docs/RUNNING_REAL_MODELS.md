# Running with real models & real data

A practical, copy-paste guide for taking ZhongJing-TCM off the offline mock and
pointing it at (a) a real LLM endpoint and (b) your own corpus. Everything here
maps to actual code — file and symbol references are given so you can verify.

The whole repo runs **offline by default** (`model="mock"`), which is how the 144
unit tests and every `run.py` demo work without an API key. This document is only
about the two switches that turn that into a real run:

1. **A real model** — set a provider + key (`src/llm_client.py`).
2. **Real data** — drop documents in `data/raw/` and run the pipeline
   (`src/m1_ingest.py` → `src/m7_assemble.py`), which produces the dataset every
   evaluation command reads (`data/final/zhongjing_tcm_full.jsonl`).

---

## 1. Connect a real model

Provider selection lives in one place — `LLMClient.__init__` /
`get_client()` in [`src/llm_client.py`](../src/llm_client.py) — and is resolved
with this precedence:

```
ZHONGJING_LLM_PROVIDER (env)  >  llm.provider (configs/pipeline.yaml)  >  "openai"
```

A model name that starts with `mock` always forces the offline provider
(`_resolve_provider`), so `--model mock` works even when a real provider is
configured — handy for a quick dry run.

### 1a. OpenAI (or any OpenAI-compatible endpoint)

```bash
export OPENAI_API_KEY=sk-...
export ZHONGJING_LLM_PROVIDER=openai          # also the default
python run.py evaluate --model gpt-4o
```

The client uses the official `openai` SDK, so **any OpenAI-compatible server**
works by overriding the base URL — vLLM, DeepSeek, Together, Groq, SGLang,
Ollama (`/v1`), etc.:

```bash
export OPENAI_API_KEY=sk-...                  # or a dummy value for local servers
export OPENAI_BASE_URL=http://localhost:8000/v1   # your vLLM / Ollama / DeepSeek URL
python run.py evaluate --model Qwen2.5-72B-Instruct
```

`OPENAI_BASE_URL` is read in `_ensure_openai()`; you can also set `llm.base_url`
in `configs/pipeline.yaml` instead of the env var.

### 1b. MiniMax (built-in)

MiniMax's endpoint is OpenAI-compatible and wired in specially so you don't have
to set the base URL yourself:

```bash
export MINIMAX_API_KEY=...
export ZHONGJING_LLM_PROVIDER=minimax
python run.py generate --concurrency 8        # batch generation, concurrent + resumable
```

When `provider=minimax`, `__init__` auto-fills:

| setting | value | override with |
|---|---|---|
| base URL | `https://api.minimaxi.com/v1` | `MINIMAX_BASE_URL` |
| API key env | `MINIMAX_API_KEY` | — |
| default model | `MiniMax-M3` | `MINIMAX_MODEL` or `--model` |

For a one-click cloud run, [`notebooks/colab_minimax_generation.ipynb`](../notebooks/colab_minimax_generation.ipynb)
drives the full pipeline with MiniMax concurrency, a live progress bar and
Drive-persisted checkpoint/resume.

### 1c. Tuning knobs (`llm` section of `configs/pipeline.yaml`)

| key | default | what it controls |
|---|---|---|
| `temperature` | `0.2` | sampling temperature (see calibration caveat in §4) |
| `max_tokens` | `8192` | output cap — large enough for long explanations / short-answer references |
| `timeout` | `60` | per-request timeout (seconds) |
| `max_concurrency` | `4` | async fan-out width for M5 generation |
| `use_cache` | `true` | on-disk response cache (`.cache/llm` via `diskcache`) |

Caching is keyed on `(provider, model, system, prompt, temperature, max_tokens)`
(`_cache_key`), so re-running an evaluation is free and deterministic. Delete
`.cache/llm` to force fresh calls.

---

## 2. Connect real data

### 2a. The pipeline that builds the dataset

Drop raw articles into `data/raw/` and run the synthesis pipeline. Each stage
writes an artefact the next one reads:

```
data/raw/*.{txt,md,html,htm,docx}
   │  M1 ingest  (src/m1_ingest.py — multi-format read + de-dup + filename parse)
   ▼
data/interim/articles.jsonl
   │  M2 quality (src/m2_quality.py — TCM-density + LLM judge gate)
   ▼  M3 topics  (src/m3_topic.py — BERTopic)   M4 label (src/m4_label.py — 9 categories)
data/interim/passages_labeled.jsonl
   │  M5 generate (src/m5_generate.py — LLM, concurrent + --resume)
   ▼
data/interim/questions_raw.jsonl
   │  M6 DTQF    (src/m6_dtqf.py — dynamic question filter)
   ▼  M7 assemble (src/m7_assemble.py — token-count + split + dataset card)
data/final/zhongjing_tcm_full.jsonl          ← every eval command reads this
data/final/zhongjing_tcm_diagnostic.jsonl    ← 10% diagnostic split (preferred by M8)
```

Run it end-to-end, or stage by stage:

```bash
export OPENAI_API_KEY=sk-...                  # or MiniMax / a local endpoint
python run.py pipeline                        # M1 → M7 in one shot
# …or per stage:
python run.py ingest                          # M1 + M2
python run.py topics && python run.py label   # M3 + M4
python run.py generate --concurrency 8        # M5 (resumable: add --resume)
python run.py dtqf && python run.py assemble  # M6 + M7
```

> M3 (BERTopic) needs the heavier NLP stack (`bertopic`, `sentence-transformers`,
> `umap-learn`, `hdbscan`) from `requirements.txt`. If you only want to exercise
> generation/evaluation, you can supply your own `passages_labeled.jsonl`
> (schema: `src/schemas.py::Passage`) and skip M3/M4.

### 2b. Input format & messy filenames

`read_document()` in `src/m1_ingest.py` handles `.txt`, `.md`, `.html`/`.htm`
(BeautifulSoup, with a regex fallback) and `.docx` (python-docx, with a zip/XML
fallback). `parse_filename()` extracts `(account, title, date)` from real-world
names like:

```
[中医书友会] - 2023-03-10 有多少大夫，正拿着"中医"的金饭碗讨饭.第22期.docx
```

— leading bracketed account, an embedded `YYYY-MM-DD`, arbitrary leading
symbols, and the title taken from the remainder. The publish date is carried
onto each `Article` so M2/M9 can use it.

### 2c. Output language

Generated questions default to **Simplified Chinese** — the v5 generation prompt
carries a top-priority `【输出语言】` block, and the language is injected from
`generate.language: 简体中文` (`configs/pipeline.yaml`), so even an English source
article yields 中文 stems/options/explanations. Override that key to target a
different language (note: the prompts themselves are authored in Chinese, so for a
non-Chinese benchmark you would also localise `prompts/gen_question.v5.txt`).

### 2d. The data contract

Every record in `data/final/*.jsonl` follows `Question` in
[`src/schemas.py`](../src/schemas.py): `single_choice` / `multiple_response` /
`short_answer` × `basic` / `intermediate` / `advanced`, with `options`,
`answer`, `explanation`, per-span token counts and a `qc_passed` flag. If you
bring a dataset from elsewhere, conform to that schema and the evaluation
commands below will consume it unchanged.

---

## 3. Which command needs which data

The evaluation commands fall into three groups by **where they get their items**.
This matters when moving from the offline demo to a real study.

### Group A — driven by your real dataset (no extra authoring)

These read `data/final/` (or `data/interim/`) directly once the pipeline has run:

| command | reads | tier/layer |
|---|---|---|
| `python run.py evaluate --model M` | `data/final/zhongjing_tcm_{diagnostic,full}.jsonl` | T0 / L1 |
| `python run.py invariance --model M` | same dataset | T0 robustness |
| `python run.py counterfactual --model M` | `data/interim/passages_labeled.jsonl` | T1 |

`evaluate` prefers the diagnostic split if present, else the full set
(`_load_dataset`). `invariance` perturbs whatever `evaluate` would load
(option-order shuffle + label-symbol relabel A–D↔甲乙丙丁/1–4). `counterfactual`
generates minimal pairs from your labelled passages with the configured
generation model.

### Group B — interactive tiers (ship with demos; swap in real cases for a study)

These run a model-vs-environment loop. They ship with small built-in demo cases
so the command works offline, but for a real evaluation you replace the demo set
with expert-authored cases (see §3.1):

| command | demo source | what to replace |
|---|---|---|
| `python run.py consult` | `t2_patient_sim.demo_cases()` | `ClinicalCase`s (hidden 证型 + findings) |
| `python run.py tools` | `t3_tools` demo tasks | `ToolTask`s (gold contraindication/dose verdicts) |
| `python run.py episode` | `t4_longitudinal.demo_episodes()` | `LongitudinalEpisode` transition graphs |
| `python run.py mdt` | `t5_mdt.demo_cases()` | `MDTCase`s (+ red-flag flag) |
| `python run.py dialogue` | `t6_dialogue.demo_cases()` | `DialogueCase`s + rubric |

### Group C — judging / measurement controls (demos illustrate the method)

These demonstrate a scoring *method* on synthetic samples; point them at real
responses/rubrics for a study:

| command | demonstrates |
|---|---|
| `python run.py process` | L2 step-level process preference + result/process gate |
| `python run.py rubric` | L3/L4 weighted rubric grade + judge meta-eval (κ vs physician labels) |
| `python run.py abstain` | A@D abstention probes (answerable vs decisive-feature-removed twins) |
| `python run.py calibrate` | ECE / Brier / reliability bins |
| `python run.py judges` | heterogeneous / tool-grounded judging (breaks shared blind spots) |

### 3.1 Replacing a demo set with real cases

Each interactive module exposes a typed case class and a `run(...)`/`evaluate_*`
entry point. To run a real study you build a list of those cases (from your own
JSONL, a spreadsheet, expert authoring, …) and call the module's `evaluate_*`
function directly instead of the demo-backed `run()`. For example, T2:

```python
import sys; sys.path.insert(0, "src")
from t2_patient_sim import ClinicalCase, PatientSim, llm_expert, evaluate_consultation

cases = [ClinicalCase(...), ...]                 # your expert-authored cases
expert = llm_expert(model="gpt-4o")              # the model under test
metrics, results = evaluate_consultation(cases, expert)  # accuracy / turns / premature-closure / abstention / recall
print(metrics)
```

The same shape applies to T3 (`ToolTask` + `run_tool_episode`), T4
(`LongitudinalEpisode` + `run_episode`), T5 (`MDTCase` + `run_mdt_case`), T6
(`DialogueCase` + `evaluate_dialogues`), and the L2/L3 rubric judges.

---

## 4. Caveats for a *valid* real run

- **Calibration needs sampling temperature.** ECE/Brier (`run.py calibrate`)
  only mean something if the model is allowed to express varied confidence. The
  default `temperature: 0.2` is tuned for deterministic generation; raise it (or
  pass `temperature` through) when measuring calibration.
- **Same-model judge = shared blind spot.** If the grader and the solver are the
  same base model, high agreement is *spurious* (`judges.judge_agreement`
  quantifies this). Use `run.py judges` (tool-grounded + heterogeneous ensemble)
  for any safety-critical scoring.
- **Demo cases are illustrative, not a benchmark.** The Group B/C built-ins exist
  so every command runs offline and is unit-tested; they are intentionally small.
  A published score requires expert-authored cases and rubrics (this is the
  "future work" called out in [`CLINICAL_EVAL_FRAMEWORK.md`](CLINICAL_EVAL_FRAMEWORK.md)).
- **Contamination control.** For headline numbers, keep a private held-out split
  out of any training/Drive folder the model provider can see.
- **Cost.** Caching (`use_cache: true`) makes re-runs free; first runs cost
  `n_runs × |dataset|` calls for `evaluate` (default `n_runs: 3`). Lower
  `evaluate.n_runs` or `max_tokens` to economise.

---

## 5. 30-second smoke test

Confirm wiring before spending tokens — `mock` exercises every code path with no
key or network:

```bash
ZHONGJING_LLM_PROVIDER=mock python run.py judges        # deterministic offline demo
make test                                               # 144 offline unit tests
```

Then swap to a real provider on a tiny slice:

```bash
export OPENAI_API_KEY=sk-...
python run.py evaluate --model gpt-4o                    # reads data/final/*, writes results/metrics.csv
```

If `evaluate` complains the dataset is missing, you haven't built it yet — run
`python run.py pipeline` (§2) first.
