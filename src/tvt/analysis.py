"""Error analysis: where does each model fail?

Slices the test set along linguistically motivated axes (text length, negation) and extracts
the examples on which the two models disagree, for qualitative inspection.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .evaluation import accuracy, macro_f1

NEGATION_PATTERNS = {
    # French negation is mostly "ne ... pas/jamais/plus/rien/aucun"; "n'" covers elisions ("n'est", "n'a").
    "fr": r"\b(ne|n'|pas|jamais|plus|rien|aucun|aucune|ni|sans)\b",
    "en": r"\b(not|no|never|nothing|none|nor|neither|without|cannot)\b|n't\b",
}


def has_negation(texts: pd.Series, language: str) -> np.ndarray:
    pattern = re.compile(NEGATION_PATTERNS.get(language, NEGATION_PATTERNS["en"]), re.IGNORECASE)
    return texts.map(lambda t: bool(pattern.search(t))).to_numpy()


def length_buckets(texts: pd.Series, n_buckets: int = 4) -> pd.Series:
    """Quantile buckets of word count, labelled like '35–71 mots' (ordered categorical)."""
    words = texts.str.split().str.len()
    edges = np.unique(np.quantile(words, np.linspace(0, 1, n_buckets + 1)).astype(int))
    labels = [f"{lo + (i > 0)}–{hi} mots" for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    return pd.cut(words, edges, labels=labels, include_lowest=True).rename("length_words")


def slice_metrics(test: pd.DataFrame, preds: dict[str, np.ndarray], language: str) -> pd.DataFrame:
    """Accuracy and macro-F1 of every model on each slice of the test set."""
    y = test["label"].to_numpy()
    buckets = length_buckets(test["text"])
    slices = {str(b): (buckets == b).to_numpy() for b in buckets.cat.categories}
    neg = has_negation(test["text"], language)
    slices["avec négation"] = neg
    slices["sans négation"] = ~neg

    rows = []
    for name, mask in slices.items():
        if mask.sum() == 0:
            continue
        row = {"slice": name, "n": int(mask.sum())}
        for model, pred in preds.items():
            row[f"{model}_accuracy"] = accuracy(y[mask], pred[mask])
            row[f"{model}_macro_f1"] = macro_f1(y[mask], pred[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements(
    test: pd.DataFrame,
    label_names: list[str],
    pred_a: np.ndarray,
    probs_a: np.ndarray,
    pred_b: np.ndarray,
    probs_b: np.ndarray,
    names: tuple[str, str] = ("baseline", "transformer"),
    per_group: int = 15,
) -> pd.DataFrame:
    """Most confident errors of each model where the other one is right, plus examples both get wrong
    with high confidence (often label noise or genuinely ambiguous texts)."""
    y = test["label"].to_numpy()
    a_ok, b_ok = pred_a == y, pred_b == y
    groups = {
        f"only {names[1]} correct": (~a_ok & b_ok, probs_a.max(axis=1)),
        f"only {names[0]} correct": (a_ok & ~b_ok, probs_b.max(axis=1)),
        "both wrong": (~a_ok & ~b_ok, (probs_a.max(axis=1) + probs_b.max(axis=1)) / 2),
    }
    frames = []
    for group, (mask, confidence) in groups.items():
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-confidence[idx])][:per_group]
        frames.append(
            pd.DataFrame(
                {
                    "group": group,
                    "test_idx": idx,
                    "label": [label_names[i] for i in y[idx]],
                    f"{names[0]}_pred": [label_names[i] for i in pred_a[idx]],
                    f"{names[0]}_conf": probs_a[idx].max(axis=1).round(3),
                    f"{names[1]}_pred": [label_names[i] for i in pred_b[idx]],
                    f"{names[1]}_conf": probs_b[idx].max(axis=1).round(3),
                    "text": test["text"].to_numpy()[idx],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
