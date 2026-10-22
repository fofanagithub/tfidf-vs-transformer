"""Metrics and statistical tests.

- point metrics: accuracy, macro-F1, ECE (calibration), per-class report
- uncertainty: percentile bootstrap confidence intervals over test examples
- paired comparison: McNemar exact test + paired bootstrap on the macro-F1 difference
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

Metric = Callable[[np.ndarray, np.ndarray], float]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


METRICS: dict[str, Metric] = {"accuracy": accuracy, "macro_f1": macro_f1}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins (Guo et al., 2017)."""
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Metric,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None,
    label_names: list[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out: dict = {}
    for name, fn in METRICS.items():
        lo, hi = bootstrap_ci(y_true, y_pred, fn, n_bootstrap, seed=seed)
        out[name] = fn(y_true, y_pred)
        out[f"{name}_ci95"] = [lo, hi]
    if probs is not None:
        out["ece"] = expected_calibration_error(np.asarray(probs), y_true)
    labels = list(range(len(label_names)))
    out["per_class"] = classification_report(
        y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0
    )
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar test on paired predictions. Exact binomial version when discordant pairs < 50."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    b = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if n < 50:
        p = binomtest(min(b, c), n, 0.5).pvalue
        return {"a_only_correct": b, "b_only_correct": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {
        "a_only_correct": b,
        "b_only_correct": c,
        "statistic": float(stat),
        "p_value": float(chi2.sf(stat, df=1)),
        "method": "chi2_continuity",
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Metric = macro_f1,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict:
    """Bootstrap distribution of metric(B) - metric(A), resampling the same examples for both models."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], pred_b[idx]) - metric(y_true[idx], pred_a[idx])
    observed = metric(y_true, pred_b) - metric(y_true, pred_a)
    # Two-sided p-value: how often the resampled difference crosses zero. It cannot be resolved below
    # 1 / n_resamples, so it is floored there (reported as "<" that value) instead of an impossible 0.
    p = min(1.0, max(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)), 1 / n_resamples))
    return {
        "diff": float(observed),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "p_value": float(p),
    }
