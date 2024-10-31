# ZhongJing-TCM-Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/abs/)

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

## Dataset Structure

```
ZhongJing-TCM-Benchmark/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── metadata/
│   ├── categories.json
│   └── topics.json
└── evaluation/
    ├── metrics/
    └── baselines/
```

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

```python
from zhongjing_tcm import TCMDataset

# Load the dataset
dataset = TCMDataset(split='train')

# Get a sample question
question = dataset[0]
print(question.text)
print(question.options)
print(question.answer)
print(question.explanation)
```

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
