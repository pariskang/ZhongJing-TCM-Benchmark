"""Unified pydantic data models for the ZhongJing-TCM benchmark pipeline.

These models are the contract between the M1–M9 modules.  Every interim
artefact (``data/interim/*.jsonl``) and the final dataset (``data/final/*.jsonl``)
is a stream of one of these records serialised with :func:`utils.save_jsonl`.

The field layout follows the engineering manual (§0.5).  A handful of optional
fields are added so the pipeline can run end-to-end (article quality scores from
M2, evaluation rows from M8); they default to ``None``/empty and therefore keep
older artefacts loadable.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Categories (9 TCM domains)                                                    #
# --------------------------------------------------------------------------- #


class Category(str, Enum):
    """The nine TCM categories used throughout the benchmark.

    Inherits from ``str`` so that comparisons against the raw Chinese label
    (e.g. ``q.category == "常见病辨证论治"`` or membership tests in M6) work
    regardless of whether the value is the enum member or a plain string loaded
    from JSON.
    """

    COMMON_DISEASE = "常见病辨证论治"
    BASIC_THEORY = "中医基础理论"
    CLASSIC_FORMULA = "经典方剂"
    DIAGNOSIS = "诊断方法与技术"
    PHARMACOLOGY = "药理与方剂组成"
    SEASONAL = "时令病与养生"
    ACUPUNCTURE = "腧穴与针灸"
    GYNECOLOGY = "妇科病与调理"
    PHYSICIAN_DEV = "医师发展与收入"

    @property
    def english(self) -> str:
        """English label (useful for plot axes / English-language reports)."""
        return _CATEGORY_EN[self]

    @classmethod
    def from_label(cls, label: "str | Category | None") -> "Optional[Category]":
        """Coerce a Chinese/English label (or enum) into a :class:`Category`."""
        if label is None:
            return None
        if isinstance(label, cls):
            return label
        text = str(label).strip()
        for member in cls:
            if text == member.value or text == member.name or text == member.english:
                return member
        return None


_CATEGORY_EN = {
    Category.COMMON_DISEASE: "Analysis & Treatment of Common Diseases",
    Category.BASIC_THEORY: "Basic TCM Theories",
    Category.CLASSIC_FORMULA: "Classical Herbal Formulas",
    Category.DIAGNOSIS: "Diagnostic Methods & Techniques",
    Category.PHARMACOLOGY: "Pharmacology & Prescription Ingredients",
    Category.SEASONAL: "Seasonal Diseases & Health Cultivation",
    Category.ACUPUNCTURE: "Acupoints & Acupuncture-Moxibustion",
    Category.GYNECOLOGY: "Gynecological Diseases & Conditioning",
    Category.PHYSICIAN_DEV: "Physician Development & Income",
}

#: All category labels in declaration order — handy for stratification.
ALL_CATEGORIES: list[str] = [c.value for c in Category]

#: Categories whose questions must contain a complete clinical presentation
#: (symptoms / signs).  Enforced by the M6 completeness check.
CLINICAL_CATEGORIES: tuple[str, ...] = (
    Category.COMMON_DISEASE.value,
    Category.GYNECOLOGY.value,
)

QuestionType = Literal["single_choice", "multiple_response", "short_answer"]
Difficulty = Literal["basic", "intermediate", "advanced"]

QUESTION_TYPES: tuple[QuestionType, ...] = (
    "single_choice",
    "multiple_response",
    "short_answer",
)
DIFFICULTIES: tuple[Difficulty, ...] = ("basic", "intermediate", "advanced")


# --------------------------------------------------------------------------- #
# Records                                                                       #
# --------------------------------------------------------------------------- #


class QualityScore(BaseModel):
    """Three-dimensional article quality score produced by M2 (LLM-as-judge)."""

    professionalism: float
    popularization: float
    practicality: float
    overall: float = 0.0
    reason: Optional[str] = None

    def recompute_overall(self) -> "QualityScore":
        self.overall = (
            self.professionalism + self.popularization + self.practicality
        ) / 3.0
        return self


class Article(BaseModel):
    """A single cleaned WeChat-public-account article (output of M1)."""

    article_id: str
    source_file: str
    account: Optional[str] = None
    title: Optional[str] = None
    publish_date: Optional[str] = None
    raw_text: str
    clean_text: str
    char_count: int
    lang: str = "zh"

    # --- Added by M2 (quality scoring & gating) ---------------------------- #
    tcm_density: Optional[float] = None
    heuristic_passed: Optional[bool] = None
    quality: Optional[QualityScore] = None
    quality_passed: Optional[bool] = None


class Passage(BaseModel):
    """A knowledge unit chunked from an article (M3) and labelled in M4."""

    passage_id: str
    article_id: str
    text: str
    topic_id: Optional[int] = None
    topic_keywords: list[str] = Field(default_factory=list)
    category: Optional[Category] = None


class Question(BaseModel):
    """A generated benchmark question (M5) filtered by DTQF (M6)."""

    question_id: str
    source_passage_id: str
    category: Category
    topic_id: int
    type: QuestionType
    difficulty: Difficulty
    stem: str
    options: dict[str, str] = Field(default_factory=dict)  # empty for short answer
    answer: list[str] = Field(default_factory=list)        # ["A"] / ["A","C"] / []
    reference_answer: Optional[str] = None
    explanation: str
    theoretical_basis: Optional[str] = None
    tokens: dict[str, int] = Field(default_factory=dict)   # stem / answer / explanation
    qc_passed: bool = False                                 # DTQF result

    def is_choice(self) -> bool:
        return self.type in ("single_choice", "multiple_response")


class EvalRecord(BaseModel):
    """One model's prediction on one question (M8) — feeds the M9 statistics."""

    question_id: str
    model: str
    category: Category
    difficulty: Difficulty
    type: QuestionType
    gold: list[str] = Field(default_factory=list)
    pred: list[str] = Field(default_factory=list)
    refused: bool = False
    correct: bool = False
    answer_tokens: int = 0          # tokens in the model's *answer* span
    output_tokens: int = 0          # tokens in the full model output
    raw_output: Optional[str] = None
