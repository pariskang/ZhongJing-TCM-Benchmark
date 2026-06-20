"""Tests for M9 — Algorithm 1 dynamic-programming token segmentation."""
import numpy as np

from m9_stats import optimal_segmentation, segment_mse


def test_dp_segmentation_recovers_known_breakpoint():
    # slope changes at x=80 → the optimal 2-segment split should land near 80
    tokens = np.arange(0, 160)
    acc = np.where(tokens < 80, 0.5 - 0.001 * tokens, 0.4 - 0.002 * (tokens - 80))
    pts = optimal_segmentation(tokens, acc, 2)
    assert len(pts) == 1
    assert 70 <= pts[0] <= 90


def test_dp_three_segments():
    tokens = np.arange(0, 150)
    acc = np.piecewise(
        tokens.astype(float),
        [tokens < 50, (tokens >= 50) & (tokens < 100), tokens >= 100],
        [lambda x: 0.3 + 0.004 * x, lambda x: 0.5 - 0.002 * (x - 50), lambda x: 0.4 + 0.003 * (x - 100)],
    )
    pts = optimal_segmentation(tokens, acc, 3)
    assert len(pts) == 2
    assert any(40 <= p <= 60 for p in pts)
    assert any(90 <= p <= 110 for p in pts)


def test_segment_mse_perfect_line_is_zero():
    x = np.arange(10, dtype=float)
    y = 2 * x + 1
    assert segment_mse(x, y, 0, 10) < 1e-9


def test_segment_mse_matches_sklearn():
    rng = np.random.default_rng(0)
    x = np.arange(20, dtype=float)
    y = rng.random(20)
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    sk = mean_squared_error(y, LinearRegression().fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1)))
    assert abs(segment_mse(x, y, 0, 20) - sk) < 1e-9


def test_handles_short_input():
    assert optimal_segmentation(np.array([1.0]), np.array([0.5]), 2) == []
