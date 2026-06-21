"""Tests for M4 category mapping & inter-annotator agreement."""
import uuid

import pytest

from m4_label import build_topic_cards, load_anchors, review_agreement, suggest_category
from schemas import Category, Passage


def _anchors():
    return load_anchors()


def test_suggest_category_hits():
    anchors = _anchors()
    assert suggest_category(["痛经", "月经", "调经"], anchors) is Category.GYNECOLOGY
    assert suggest_category(["针灸", "取穴", "艾灸"], anchors) is Category.ACUPUNCTURE
    assert suggest_category(["桂枝汤", "方解", "加减"], anchors) is Category.CLASSIC_FORMULA


def test_suggest_category_no_hit():
    assert suggest_category(["足球", "篮球"], _anchors()) is None


def test_build_topic_cards():
    passages = [
        Passage(passage_id=str(uuid.uuid4()), article_id="a", text="月经不调痛经调理" * 5,
                topic_id=0, topic_keywords=["痛经", "月经", "调经"]),
        Passage(passage_id=str(uuid.uuid4()), article_id="a", text="痛经的辨证论治" * 5,
                topic_id=0, topic_keywords=["痛经", "月经", "调经"]),
    ]
    cards = build_topic_cards(passages, _anchors())
    assert len(cards) == 1
    assert cards[0]["suggested_category"] == Category.GYNECOLOGY.value
    assert cards[0]["size"] == 2


def test_review_agreement_pass():
    a = ["经典方剂", "腧穴与针灸", "妇科病与调理", "中医基础理论", "经典方剂"]
    b = ["经典方剂", "腧穴与针灸", "妇科病与调理", "中医基础理论", "诊断方法与技术"]
    k = review_agreement(a, b, threshold=0.5)
    assert 0.5 <= k <= 1.0


def test_review_agreement_assert_fails_on_low_kappa():
    a = ["经典方剂"] * 3 + ["妇科病与调理"] * 3
    b = ["妇科病与调理"] * 3 + ["经典方剂"] * 3  # perfectly disagreeing
    with pytest.raises(AssertionError):
        review_agreement(a, b, threshold=0.8)
