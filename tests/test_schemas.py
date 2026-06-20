"""Tests for the pydantic data models."""
from schemas import ALL_CATEGORIES, Category, Question


def test_nine_categories():
    assert len(list(Category)) == 9
    assert len(ALL_CATEGORIES) == 9


def test_str_enum_equality():
    # Category is a str-enum: must compare equal to its raw Chinese label.
    assert Category.COMMON_DISEASE == "常见病辨证论治"
    assert Category.GYNECOLOGY in ("常见病辨证论治", "妇科病与调理")


def test_from_label():
    assert Category.from_label("经典方剂") is Category.CLASSIC_FORMULA
    assert Category.from_label("CLASSIC_FORMULA") is Category.CLASSIC_FORMULA
    assert Category.from_label(Category.DIAGNOSIS) is Category.DIAGNOSIS
    assert Category.from_label("not-a-category") is None
    assert Category.from_label(None) is None


def test_english_label():
    assert "Acup" in Category.ACUPUNCTURE.english


def test_question_roundtrip():
    q = Question(
        question_id="q1",
        source_passage_id="p1",
        category="经典方剂",
        topic_id=3,
        type="single_choice",
        difficulty="basic",
        stem="x" * 20,
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        answer=["A"],
        explanation="y" * 25,
    )
    dumped = q.model_dump(mode="json")
    assert dumped["category"] == "经典方剂"        # enum serialised to value
    assert Question.model_validate(dumped) == q
    assert q.is_choice()
