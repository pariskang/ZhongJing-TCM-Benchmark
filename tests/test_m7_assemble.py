"""Tests for M7 assembly: token counting, stratified split, dataset card."""
from collections import Counter

from m7_assemble import count_tokens, dataset_card, stratified_sample


def _grid(make_q, per_cell=4):
    qs = []
    for cat in ("经典方剂", "中医基础理论", "腧穴与针灸"):
        for diff in ("basic", "intermediate", "advanced"):
            for typ in ("single_choice", "multiple_response", "short_answer"):
                for _ in range(per_cell):
                    qs.append(make_q(category=cat, difficulty=diff, type=typ,
                                     topic_id=hash(cat) % 5))
    return qs


def test_count_tokens_populates_three_spans(make_q):
    q = count_tokens(make_q())
    assert set(q.tokens) == {"stem", "answer", "explanation"}
    assert all(v >= 0 for v in q.tokens.values())
    assert q.tokens["stem"] > 0


def test_stratified_sample_covers_all_cells(make_q):
    qs = _grid(make_q, per_cell=4)
    sample = stratified_sample(qs, frac=0.25, seed=0)
    # 3 categories x 3 difficulties x 3 types = 27 cells, ≥1 each
    cells = Counter((q.category, q.difficulty, q.type) for q in sample)
    assert len(cells) == 27
    assert all(v >= 1 for v in cells.values())


def test_dataset_card(make_q):
    qs = [count_tokens(q) for q in _grid(make_q, per_cell=2)]
    card = dataset_card(qs)
    assert set(card.columns) == {"NoT", "N", "Tokens"}
    assert card["N"].sum() == len(qs)
