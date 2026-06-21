"""M9 — Statistical analysis. ★ includes the DP token-segmentation

Reproduces the paper's statistics:

* two-way ANOVA (Category × LLM) on answer-token length + Tukey HSD,
* accuracy-vs-token-length regression (linear / quadratic / cubic / log),
* the dynamic-programming optimal token segmentation (Algorithm 1) that recovers
  per-model break-points (e.g. GPT-4's 0–79-token regime).
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from config import Config, load_config
from schemas import EvalRecord, Question
from utils import ensure_parent, get_logger, iter_jsonl, load_jsonl_as

_log = get_logger("m9_stats")


# --------------------------------------------------------------------------- #
# 9.1 — ANOVA + Tukey HSD                                                       #
# --------------------------------------------------------------------------- #


def anova_answer_tokens(df):
    """Two-way ANOVA ``answer_tokens ~ C(category) * C(model)`` + Tukey on model."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    n_models = df["model"].nunique()
    n_cats = df["category"].nunique()
    if n_models >= 2 and n_cats >= 2:
        formula = "answer_tokens ~ C(category) + C(model) + C(category):C(model)"
    elif n_cats >= 2:
        formula = "answer_tokens ~ C(category)"
    elif n_models >= 2:
        formula = "answer_tokens ~ C(model)"
    else:
        _log.warning("ANOVA needs ≥2 levels; returning means only")
        return df.groupby(["model", "category"])["answer_tokens"].mean().reset_index(), None

    model = smf.ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)
    tukey = None
    if n_models >= 2:
        tukey = pairwise_tukeyhsd(df["answer_tokens"], df["model"], alpha=0.05)
    return table, tukey


# --------------------------------------------------------------------------- #
# 9.2 — Regression (accuracy vs token length)                                   #
# --------------------------------------------------------------------------- #


def accuracy_by_token_bin(tokens, correct, n_bins: int = 12):
    """Bin questions by token length; return (bin_center, mean_accuracy) arrays."""
    tokens = np.asarray(tokens, float)
    correct = np.asarray(correct, float)
    if len(tokens) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(tokens.min(), tokens.max() + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(tokens, edges) - 1, 0, n_bins - 1)
    centers, accs = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        centers.append((edges[b] + edges[b + 1]) / 2)
        accs.append(correct[mask].mean())
    return np.array(centers), np.array(accs)


def fit_curves(token_len, accuracy) -> dict:
    """Fit linear / quadratic / cubic / log curves; report R² and MSE."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures

    x = np.asarray(token_len, float).reshape(-1, 1)
    y = np.asarray(accuracy, float)
    res: dict[str, dict] = {}
    if len(y) < 2:
        return res
    for name, deg in [("linear", 1), ("poly2", 2), ("poly3", 3)]:
        Xp = PolynomialFeatures(deg).fit_transform(x)
        m = LinearRegression().fit(Xp, y)
        pred = m.predict(Xp)
        res[name] = dict(R2=float(r2_score(y, pred)), MSE=float(mean_squared_error(y, pred)))
    xl = np.log(x + 1)
    m = LinearRegression().fit(xl, y)
    pred = m.predict(xl)
    res["log"] = dict(R2=float(r2_score(y, pred)), MSE=float(mean_squared_error(y, pred)))
    return res


# --------------------------------------------------------------------------- #
# 9.3 — Algorithm 1: dynamic-programming optimal token segmentation             #
# --------------------------------------------------------------------------- #


def segment_mse(tokens, correctness, i: int, j: int) -> float:
    """MSE of the least-squares line over the half-open segment ``[i, j)``.

    Closed-form OLS (identical to ``LinearRegression`` but far faster, since the
    DP calls this O(k·n²) times).
    """
    if j - i < 2:
        return 0.0
    x = np.asarray(tokens[i:j], float)
    y = np.asarray(correctness[i:j], float)
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0.0:                       # degenerate x → best fit is ȳ
        return float(((y - ym) ** 2).mean())
    b = float(((x - xm) * (y - ym)).sum()) / sxx
    a = ym - b * xm
    pred = a + b * x
    return float(((y - pred) ** 2).mean())


def optimal_segmentation(tokens, correctness, num_segments: int) -> list[int]:
    """Algorithm 1 — break-points minimising total per-segment linear-fit MSE."""
    tokens = np.asarray(tokens, float)
    correctness = np.asarray(correctness, float)
    order = np.argsort(tokens)
    tokens, correctness = tokens[order], correctness[order]
    n = len(tokens)
    if n == 0 or num_segments < 1:
        return []
    num_segments = min(num_segments, n)

    dp = np.full((num_segments + 1, n + 1), np.inf)
    split = np.zeros((num_segments + 1, n + 1), dtype=int)
    for j in range(1, n + 1):                                   # base case: 1 segment
        dp[1][j] = segment_mse(tokens, correctness, 0, j)
    for i in range(2, num_segments + 1):                        # recurrence
        for j in range(i, n + 1):
            for k in range(i - 1, j):
                cost = dp[i - 1][k] + segment_mse(tokens, correctness, k, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    points, j = [], n                                           # back-track
    for i in range(num_segments, 0, -1):
        points.append(j)
        j = split[i][j]
    # points = [n, b_{k-1}, …, b_1, (0)] — the internal break-points are points[1:]
    # (drop the right edge ``n``); the implicit left edge 0 is filtered out below.
    return sorted(set(p for p in points[1:] if 0 < p < n))


def per_model_segments(df, num_segments: int = 2) -> dict:
    """Per-model token break-points (token values at the optimal split indices)."""
    out: dict[str, list[float]] = {}
    for model, g in df.groupby("model"):
        tokens = g["tokens"].to_numpy(float)
        correct = g["correct"].to_numpy(float)
        idx = optimal_segmentation(tokens, correct, num_segments)
        sorted_tokens = np.sort(tokens)
        out[str(model)] = [float(sorted_tokens[i]) for i in idx]
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def _build_dataframe(cfg: Config):
    """Join evaluation records with question stem-token lengths into a frame."""
    import pandas as pd

    results = cfg.path("paths.results_dir")
    final = cfg.path("paths.final_dir")

    stem_tokens: dict[str, int] = {}
    full = final / "zhongjing_tcm_full.jsonl"
    if full.exists():
        for q in load_jsonl_as(full, Question):
            stem_tokens[q.question_id] = q.tokens.get("stem", 0)

    rows = []
    for path in sorted(results.glob("eval_*.jsonl")):
        for rec in (EvalRecord.model_validate(r) for r in iter_jsonl(path)):
            rows.append(
                {
                    "model": rec.model,
                    "category": rec.category.value if hasattr(rec.category, "value") else rec.category,
                    "difficulty": rec.difficulty,
                    "type": rec.type,
                    "answer_tokens": rec.answer_tokens,
                    "output_tokens": rec.output_tokens,
                    "correct": int(rec.correct),
                    "refused": int(rec.refused),
                    "tokens": stem_tokens.get(rec.question_id, rec.answer_tokens),
                }
            )
    return pd.DataFrame(rows)


def run(cfg: Optional[Config] = None) -> dict:
    """Run the full statistical battery and persist tables / figures."""
    cfg = cfg or load_config()
    results = cfg.path("paths.results_dir")
    df = _build_dataframe(cfg)
    if df.empty:
        _log.warning("no evaluation records found under %s", results)
        return {}

    # --- ANOVA + Tukey ----------------------------------------------------- #
    try:
        table, tukey = anova_answer_tokens(df)
        table.to_csv(ensure_parent(results / "anova.csv"))
        if tukey is not None:
            (results / "tukey.txt").write_text(str(tukey.summary()), encoding="utf-8")
        _log.info("ANOVA table:\n%s", table)
    except Exception as exc:  # noqa: BLE001
        _log.warning("ANOVA skipped: %s", exc)

    # --- Regression -------------------------------------------------------- #
    reg = {}
    for model, g in df.groupby("model"):
        centers, accs = accuracy_by_token_bin(g["tokens"], g["correct"])
        reg[str(model)] = fit_curves(centers, accs)
    (ensure_parent(results / "regression.json")).write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- DP segmentation --------------------------------------------------- #
    segments = per_model_segments(df, num_segments=cfg.get("stats.num_segments", 2))
    (results / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log.info("token break-points: %s", segments)

    try:
        plot_token_range_performance(df, cfg)
    except Exception as exc:  # noqa: BLE001
        _log.warning("token-range figure skipped: %s", exc)

    try:
        plot_regression_curves(df, cfg)
    except Exception as exc:  # noqa: BLE001
        _log.warning("regression-curves figure skipped: %s", exc)

    return {"anova": str(results / "anova.csv"), "segments": segments, "regression": reg}


def plot_token_range_performance(df, cfg: Optional[Config] = None):
    """Accuracy vs question-token-length curve per model (paper token-range fig)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in df.groupby("model"):
        centers, accs = accuracy_by_token_bin(g["tokens"], g["correct"])
        if len(centers):
            ax.plot(centers, accs, marker="o", label=str(model))
    ax.set_xlabel("Question token length")
    ax.set_ylabel("Accuracy")
    ax.set_title("Token-range performance")
    ax.legend()
    out = ensure_parent(cfg.path("paths.results_dir") / "figures" / "token_range_performance.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log.info("wrote %s", out)
    return out


def plot_regression_curves(df, cfg: Optional[Config] = None):
    """Accuracy-vs-token-length regression curves per model (linear/poly2/poly3/log)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    cfg = cfg or load_config()
    models = list(df["model"].unique())
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        g = df[df["model"] == model]
        centers, accs = accuracy_by_token_bin(g["tokens"].to_numpy(float), g["correct"].to_numpy(float))
        if len(centers) < 2:
            continue
        ax.scatter(centers, accs, color="black", s=30, zorder=3, label="data", alpha=0.8)
        x_fit = np.linspace(centers.min(), centers.max(), 200).reshape(-1, 1)
        for name, deg in [("linear", 1), ("poly2", 2), ("poly3", 3)]:
            Xp = PolynomialFeatures(deg).fit_transform(centers.reshape(-1, 1))
            m_reg = LinearRegression().fit(Xp, accs)
            Xf = PolynomialFeatures(deg).fit_transform(x_fit)
            ax.plot(x_fit.ravel(), m_reg.predict(Xf), label=name)
        xl = np.log(centers.reshape(-1, 1) + 1)
        m_reg = LinearRegression().fit(xl, accs)
        xfl = np.log(x_fit + 1)
        ax.plot(x_fit.ravel(), m_reg.predict(xfl), label="log", linestyle="--")
        ax.set_xlabel("Question token length")
        ax.set_ylabel("Accuracy")
        ax.set_title(str(model))
        ax.legend(fontsize=8)
    out = ensure_parent(cfg.path("paths.results_dir") / "figures" / "regression_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log.info("wrote %s", out)
    return out


if __name__ == "__main__":
    run()
