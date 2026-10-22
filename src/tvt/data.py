"""Dataset loading and splitting.

Both models consume exactly the same train / validation / test DataFrames produced here,
so any difference in results comes from the models, not from the data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig
from .utils import get_logger

log = get_logger()

_WS = re.compile(r"\s+")


@dataclass
class Splits:
    train: pd.DataFrame  # columns: text, label
    val: pd.DataFrame
    test: pd.DataFrame
    label_names: list[str]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    def summary(self) -> dict:
        def describe(df: pd.DataFrame) -> dict:
            lengths = df["text"].str.split().str.len()
            return {
                "n": len(df),
                "label_distribution": df["label"].value_counts().sort_index().to_dict(),
                "mean_words": round(float(lengths.mean()), 1),
                "median_words": float(lengths.median()),
                "p95_words": float(lengths.quantile(0.95)),
            }

        return {
            "label_names": self.label_names,
            "train": describe(self.train),
            "val": describe(self.val),
            "test": describe(self.test),
            "test_fingerprint": fingerprint(self.test),
        }


def normalize_text(text: str) -> str:
    """Minimal, model-agnostic cleaning: collapse whitespace. Casing and punctuation are kept on purpose
    (the transformer tokenizer relies on them; the TF-IDF vectorizer lowercases itself)."""
    return _WS.sub(" ", str(text)).strip()


def fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a split, used to check that every model was evaluated on the same test set."""
    h = hashlib.sha256()
    for text, label in zip(df["text"], df["label"]):
        h.update(f"{label}\t{text}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


def clean_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(text=df["text"].map(normalize_text))
    df = df[df["text"].str.len() > 0]
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def remove_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop train/val rows whose text also appears in test (and train rows duplicated in val)."""
    test_texts = set(test["text"])
    n_train, n_val = len(train), len(val)
    val = val[~val["text"].isin(test_texts)]
    train = train[~train["text"].isin(test_texts | set(val["text"]))]
    dropped = (n_train - len(train), n_val - len(val))
    if any(dropped):
        log.info("Leakage removal: dropped %d train and %d val rows also present in a later split", *dropped)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def from_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_names: list[str],
    cfg: DataConfig,
    seed: int,
    val: pd.DataFrame | None = None,
) -> Splits:
    """Build clean, leakage-free, optionally subsampled splits from raw frames with `text`/`label` columns."""
    train, test = clean_split(train), clean_split(test)
    if val is None:
        train, val = train_test_split(train, test_size=cfg.val_size, stratify=train["label"], random_state=seed)
    val = clean_split(val)
    train, val = remove_leakage(train, val, test)

    train = stratified_sample(train, cfg.max_train_samples, seed)
    val = stratified_sample(val, cfg.max_eval_samples, seed)
    test = stratified_sample(test, cfg.max_eval_samples, seed)
    return Splits(train=train, val=val, test=test, label_names=label_names)


def load_splits(cfg: DataConfig, seed: int) -> Splits:
    """Load a Hugging Face dataset and return deterministic train / val / test splits."""
    from datasets import load_dataset

    ds = load_dataset(cfg.name)
    features = ds["train"].features
    label_feature = features[cfg.label_column]
    label_names = list(getattr(label_feature, "names", None) or sorted(set(list(ds["train"][cfg.label_column]))))

    def to_frame(split: str) -> pd.DataFrame:
        frame = ds[split].to_pandas()[[cfg.text_column, cfg.label_column]]
        return frame.rename(columns={cfg.text_column: "text", cfg.label_column: "label"})

    val_split = next((s for s in ("validation", "valid", "dev") if s in ds), None)
    splits = from_frames(
        train=to_frame("train"),
        test=to_frame("test"),
        val=to_frame(val_split) if val_split else None,
        label_names=[str(n) for n in label_names],
        cfg=cfg,
        seed=seed,
    )
    log.info("Data: train=%d val=%d test=%d labels=%s", len(splits.train), len(splits.val), len(splits.test), splits.label_names)
    return splits
