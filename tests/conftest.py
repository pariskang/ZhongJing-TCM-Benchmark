"""Pytest fixtures & path setup.

Adds ``src/`` to ``sys.path`` and forces the offline ``mock`` LLM provider so the
whole suite runs without API keys or network.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Force the offline provider before any client is constructed.
os.environ["ZHONGJING_LLM_PROVIDER"] = "mock"
os.environ.setdefault("ZHONGJING_LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def _mock_llm():
    """Reset the cached default client so every test uses the mock provider."""
    import llm_client

    llm_client._DEFAULT_CLIENT = None
    yield
    llm_client._DEFAULT_CLIENT = None


@pytest.fixture
def make_q():
    """Factory for valid :class:`~schemas.Question` instances."""
    from schemas import Question

    def _make(
        category: str = "经典方剂",
        type: str = "single_choice",
        stem: str = "下列关于小柴胡汤君药的描述，哪一项是正确的？",
        options=None,
        answer=None,
        explanation: str = "柴胡为君药，疏解少阳之邪，透邪外出，是和解少阳的关键药物。",
        reference_answer=None,
        topic_id: int = 0,
        difficulty: str = "basic",
    ) -> Question:
        if type == "short_answer":
            options = {} if options is None else options
            answer = [] if answer is None else answer
            if reference_answer is None:
                reference_answer = "柴胡，疏解少阳，透邪外出。"
        else:
            options = options if options is not None else {"A": "柴胡", "B": "黄芩", "C": "半夏", "D": "甘草"}
            answer = answer if answer is not None else ["A"]
        return Question(
            question_id=str(uuid.uuid4()),
            source_passage_id="passage-0",
            category=category,
            topic_id=topic_id,
            type=type,
            difficulty=difficulty,
            stem=stem,
            options=options,
            answer=answer,
            reference_answer=reference_answer,
            explanation=explanation,
        )

    return _make


@pytest.fixture
def kw_ext():
    """A keyword extractor trained on a small TCM corpus."""
    from m6_dtqf import KeywordExtractor

    corpus = [
        ["患者", "发热", "恶寒", "舌红", "苔黄", "脉数", "辨证"],
        ["小柴胡汤", "柴胡", "黄芩", "半夏", "少阳", "和解"],
        ["脾胃", "虚弱", "食少", "纳呆", "乏力", "健脾", "益气"],
        ["痛经", "小腹", "胀痛", "经血", "瘀血", "理气", "活血"],
    ]
    return KeywordExtractor(corpus, min_count=1)
