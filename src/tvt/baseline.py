"""TF-IDF + logistic regression baseline.

The baseline is deliberately strong (word and character n-grams, sublinear TF, tuned regularisation):
a transformer only "wins" if it beats a well-tuned classical model, not a strawman.
Model selection uses the validation split only; the test split is touched once, at the end.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import BaselineConfig
from .data import Splits
from .evaluation import macro_f1
from .utils import get_logger, path_size_mb, save_json, timer

log = get_logger()

_SPEC = re.compile(r"^(word|char)_(\d+)_(\d+)$")


def build_vectorizer(feature_set: str, cfg: BaselineConfig) -> FeatureUnion:
    """`feature_set` is a '+'-joined list of `<word|char>_<min_n>_<max_n>`, e.g. 'word_1_2+char_2_5'."""
    parts = []
    for spec in feature_set.split("+"):
        match = _SPEC.match(spec)
        if not match:
            raise ValueError(f"Invalid feature spec {spec!r}")
        kind, lo, hi = match.group(1), int(match.group(2)), int(match.group(3))
        vec = TfidfVectorizer(
            analyzer="word" if kind == "word" else "char_wb",
            ngram_range=(lo, hi),
            lowercase=True,
            strip_accents=None,  # accents carry meaning in French ("où" vs "ou")
            sublinear_tf=True,
            min_df=cfg.min_df,
            max_features=cfg.max_features,
            token_pattern=r"(?u)\b\w+\b" if kind == "word" else None,  # keep 1-char tokens like "a", "y"
            dtype=np.float32,
        )
        parts.append((spec, vec))
    return FeatureUnion(parts)


def build_classifier(C: float, cfg: BaselineConfig) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=cfg.max_iter, class_weight=cfg.class_weight)


def select_model(splits: Splits, cfg: BaselineConfig) -> tuple[dict, pd.DataFrame]:
    """Grid search on the validation split. Vectorizers are fitted once per feature set (on train only)."""
    rows = []
    y_train, y_val = splits.train["label"].to_numpy(), splits.val["label"].to_numpy()
    for feature_set in cfg.feature_sets:
        vectorizer = build_vectorizer(feature_set, cfg)
        X_train = vectorizer.fit_transform(splits.train["text"])
        X_val = vectorizer.transform(splits.val["text"])
        for C in cfg.C_grid:
            with timer() as t:
                clf = build_classifier(C, cfg).fit(X_train, y_train)
            score = macro_f1(y_val, clf.predict(X_val))
            rows.append({"feature_set": feature_set, "C": C, "n_features": X_train.shape[1], "val_macro_f1": score, "fit_seconds": t["seconds"]})
            log.info("baseline | %-20s C=%-6g val macro-F1=%.4f (%.1fs)", feature_set, C, score, t["seconds"])
    grid = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    return {"feature_set": best["feature_set"], "C": float(best["C"])}, grid


def run_baseline(splits: Splits, cfg: BaselineConfig, out_dir: Path, save_model: bool = True) -> dict:
    """Tune on val, refit the best configuration on train, predict test. Writes predictions and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    best, grid = select_model(splits, cfg)
    grid.to_csv(out_dir / "val_grid.csv", index=False)
    log.info("baseline | best config: %s", best)

    pipeline = Pipeline([("tfidf", build_vectorizer(best["feature_set"], cfg)), ("clf", build_classifier(best["C"], cfg))])
    with timer() as t_fit:
        pipeline.fit(splits.train["text"], splits.train["label"])
    with timer() as t_pred:
        probs = pipeline.predict_proba(splits.test["text"])

    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    model_size_mb = path_size_mb(model_path)
    if not save_model:
        model_path.unlink()
    save_predictions(out_dir / "predictions.csv", splits.test["label"].to_numpy(), probs)

    meta = {
        "model": "tfidf+logreg",
        "best_params": best,
        "val_macro_f1": float(grid.iloc[0]["val_macro_f1"]),
        "n_features": int(grid.iloc[0]["n_features"]),
        "train_seconds": t_fit["seconds"],
        "inference_seconds": t_pred["seconds"],
        "inference_examples_per_second": len(splits.test) / t_pred["seconds"],
        "inference_device": "cpu",
        "model_size_mb": model_size_mb,
        "n_parameters": int(pipeline.named_steps["clf"].coef_.size + pipeline.named_steps["clf"].intercept_.size),
    }
    save_json(meta, out_dir / "meta.json")
    return meta


def save_predictions(path: Path, y_true: np.ndarray, probs: np.ndarray) -> None:
    frame = pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(probs.shape[1])])
    frame.insert(0, "pred", probs.argmax(axis=1))
    frame.insert(0, "label", y_true)
    frame.index.name = "test_idx"
    frame.to_csv(path)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, index_col="test_idx")
    probs = frame[[c for c in frame.columns if c.startswith("prob_")]].to_numpy()
    return frame["label"].to_numpy(), frame["pred"].to_numpy(), probs
