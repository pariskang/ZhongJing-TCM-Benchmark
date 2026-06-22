# ZhongJing-TCM-Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/ZhongJing-TCM-Benchmark/blob/main/notebooks/colab_minimax_generation.ipynb)

A comprehensive benchmark dataset for evaluating Traditional Chinese Medicine (TCM) common sense knowledge in Large Language Models.

## Overview

ZhongJing-TCM is a pioneering dataset designed to evaluate Large Language Models' proficiency in Traditional Chinese Medicine. Named after the renowned physician Zhang ZhongJing, this benchmark comprises 12,000 clinically relevant questions spanning 175 topics across 9 TCM categories, stratified into three difficulty levels.

## Key Features

- **Comprehensive Coverage**: 12,000 clinically relevant questions
- **Diverse Topics**: 175 unique topics across 9 TCM categories
- **Multiple Question Types**: Single-choice, multiple-choice, and open-ended questions
- **Difficulty Levels**: Three-tiered stratification
- **Expert Validation**: Verified by multi TCM experts
- **High-Quality Data**: Generated using innovative three-stage synthetic data generation strategy

## Repository Structure

This repository contains the full **data-synthesis & evaluation pipeline** that
produces the benchmark from raw articles — see [`docs/PIPELINE.md`](docs/PIPELINE.md).

```
ZhongJing-TCM-Benchmark/
├── configs/pipeline.yaml      # models, thresholds, paths
├── data/{raw,interim,final}/  # inputs → stage artefacts → released dataset
├── lexicons/                  # TCM term dict, stopwords, 9-category anchors
├── prompts/                   # versioned generation / judge / STAGER prompts
├── src/
│   ├── m1_ingest.py  m2_quality.py  m3_topic.py  m4_label.py
│   ├── m5_generate.py  m6_dtqf.py   m7_assemble.py
│   ├── m8_evaluate.py  m9_stats.py
│   ├── schemas.py  llm_client.py  config.py  utils.py
├── tests/                     # pytest suite (offline, mock LLM)
├── run.py                     # CLI orchestration (typer)
└── Makefile
```

The pipeline maps 1:1 to the paper: BERTopic (Eq. 1–3), 9-category labelling
(Eq. 4), the **DTQF** dynamic question filter (Eq. 5–9), and the
dynamic-programming token segmentation (Algorithm 1). See the formula↔code index
in [`docs/PIPELINE.md`](docs/PIPELINE.md).

The benchmark's evaluation standard — the POMDP view, the T0–T6 fidelity ladder
and the four orthogonal scoring layers (result / process / safety / interaction),
with each requirement mapped to a module and an implemented/planned status — is in
[`docs/CLINICAL_EVAL_FRAMEWORK.md`](docs/CLINICAL_EVAL_FRAMEWORK.md). The v5
generation prompt already enforces complete disease-course case stems and
hard-to-separate options from that standard.

## Categories

The dataset covers 9 major TCM domains:

1. Analysis and Treatment of Common Diseases
2. Classical Herbal Formulas
3. Basic TCM Theories
4. Gynecological Diseases
5. Seasonal Diseases and Health Cultivation
6. Pharmacology and Prescription Ingredients
7. Diagnostic Methods and Techniques
8. Acupoints and Acupuncture Moxibustion
9. Physician Development

## Usage

### Run the pipeline

```bash
make install                            # dependencies (see requirements.txt)
make test                               # 54 unit tests, fully offline
ZHONGJING_LLM_PROVIDER=mock make demo   # run M1→M7 with the offline mock LLM

# real generation / evaluation
export OPENAI_API_KEY=sk-...            # any OpenAI-compatible endpoint works
python run.py pipeline                  # M1..M7
python run.py evaluate --model gpt-4o   # M8
python run.py stats                     # M9 (ANOVA, regression, DP segmentation)

# robustness & interactive (clinical-eval framework — docs/CLINICAL_EVAL_FRAMEWORK.md)
python run.py invariance --model gpt-4o    # option-order & label-symbol (A–D↔甲乙丙丁/1–4) invariance
python run.py counterfactual               # T1 counterfactual minimal pairs (flip one 四诊 → answer flips)
python run.py consult --model gpt-4o       # T2 active-inquiry consultation vs the patient simulator
python run.py process --model gpt-4o       # L2 step-level process preference + result/process gate
python run.py rubric --model gpt-4o        # L3/L4 weighted rubric grading + judge meta-evaluation
python run.py abstain --model gpt-4o       # A@D abstention probes (abstain iff info insufficient)
python run.py tools --model gpt-4o         # T3 tool-use agent (contraindication checks; tool-grounding)
python run.py episode --model gpt-4o       # T4 longitudinal episode (follow-up & adjustment; 同病异治 trajectory)
```

### Batch generation with MiniMax (concurrent + resumable)

MiniMax's OpenAI-compatible endpoint is supported out of the box:

```bash
export MINIMAX_API_KEY=...                          # https://api.minimaxi.com/v1
export ZHONGJING_LLM_PROVIDER=minimax
python run.py generate --concurrency 8              # parallel question generation
python run.py generate --resume                     # re-run to fill only missing items
python run.py generate --no-progress                # hide the live progress bar
```

Generation is **real-time** on both ends: a live `tqdm` bar advances per passage
(percentage / speed / running question count), and each passage's questions are
flushed to `data/interim/questions_raw.jsonl` the instant it completes — so the
file grows as you watch and an interrupted run loses nothing (that on-disk
checkpoint is exactly what `--resume` reads back).

The output cap defaults to `llm.max_tokens: 8192` (in `configs/pipeline.yaml`),
which leaves room for long step-by-step explanations and short-answer references;
lower it to save tokens if you only need choice questions.

For a one-click cloud run, open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/ZhongJing-TCM-Benchmark/blob/main/notebooks/colab_minimax_generation.ipynb)

[`notebooks/colab_minimax_generation.ipynb`](notebooks/colab_minimax_generation.ipynb)
drives the full M1→M9 pipeline with MiniMax concurrency, a live progress bar and
checkpoint/resume (persisted to Google Drive across runtime disconnects). Step 4
exposes `MAX_TOKENS` (default 8192), `MAX_CONCURRENCY` and the model name; step 6
reads `.txt`, `.html` and `.docx` documents straight from a Google Drive folder
(e.g. `/content/drive/MyDrive/zhongjing-tcm-benchmark/yichengyoudao`) and parses
messy filenames such as `[公众号] - 2023-03-10 标题.docx` automatically.

### Load the generated questions

```python
import json

with open("data/final/zhongjing_tcm_full.jsonl", encoding="utf-8") as fh:
    questions = [json.loads(line) for line in fh]

q = questions[0]
print(q["stem"], q["options"], q["answer"], q["explanation"])
```

Each record follows the `Question` schema in [`src/schemas.py`](src/schemas.py)
(`single_choice` / `multiple_response` / `short_answer` × `basic` /
`intermediate` / `advanced`, with per-span token counts and a `qc_passed` flag).

## Contributing

We welcome contributions to improve the dataset and evaluation metrics. Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We acknowledge the contributions of ancient Chinese medicine physicians, notably ZhongJing Zhang after whom our dataset is named. Special thanks to the nonprofit organization Future Medicine Philosophy (Ful-Phil) and all collaborating physicians who contributed to this research.

## Citation

```bibtex
@article{anonymous2024zhongjing,
  title={ZhongJing-TCM: A Benchmark for Evaluating Traditional Chinese Medicine Common Sense Knowledge in Large Language Models},
  author={Anonymous},
  journal={ArXiv},
  year={2024}
}
```
