"""Data-efficiency study: test macro-F1 of both models as a function of the number of training examples.

Answers the practical question behind the benchmark: from how much labelled data does fine-tuning pay off?
Results are appended row by row to a CSV, so an interrupted run (e.g. a Colab disconnect) resumes where it stopped.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .baseline import load_predictions, run_baseline
from .config import ExperimentConfig
from .data import Splits, stratified_sample
from .evaluation import accuracy, macro_f1
from .transformer import run_transformer_seed
from .utils import get_logger

log = get_logger()


def run_learning_curve(splits: Splits, cfg: ExperimentConfig, models: tuple[str, ...] = ("baseline", "transformer")) -> pd.DataFrame:
    out_dir = cfg.run_dir / "learning_curve"
    csv_path = out_dir / "learning_curve.csv"
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=["model", "train_size", "seed"])
    done_keys = set(zip(done["model"], done["train_size"], done["seed"]))

    sizes = sorted({min(s, len(splits.train)) for s in cfg.learning_curve_sizes})
    for size in sizes:
        for seed in cfg.learning_curve_seeds:
            sub = replace(splits, train=stratified_sample(splits.train, size, seed))
            for model in models:
                if (model, size, seed) in done_keys:
                    continue
                run_dir = out_dir / model / f"n{size}_s{seed}"
                if model == "baseline":
                    meta = run_baseline(sub, cfg.baseline, run_dir, save_model=False)
                else:
                    meta = run_transformer_seed(sub, replace(cfg.transformer, save_model=False), seed, run_dir)
                y, pred, _ = load_predictions(run_dir / "predictions.csv")
                row = {
                    "model": model,
                    "train_size": size,
                    "seed": seed,
                    "test_macro_f1": macro_f1(y, pred),
                    "test_accuracy": accuracy(y, pred),
                    "train_seconds": meta["train_seconds"],
                }
                log.info("learning curve | %s n=%d seed=%d macro-F1=%.4f", model, size, seed, row["test_macro_f1"])
                pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    return pd.read_csv(csv_path)
