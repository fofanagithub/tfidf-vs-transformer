"""One-off recovery of test predictions written in length-grouped order (bug fixed in tvt.transformer).

With transformers 5.x, `train_sampling_strategy="group_by_length"` also applies to evaluation / prediction:
`trainer.predict` returned rows in the order of a seeded LengthGroupedSampler instead of dataset order.
That order is deterministic (fresh `torch.Generator().manual_seed(data_seed)`, eval batch size, token lengths),
so it can be rebuilt exactly and the saved probabilities put back in dataset order.

The recovery is self-validating: a wrong permutation yields chance-level accuracy, the right one recovers
the accuracy measured on validation during training.

    python scripts/recover_predictions.py --config configs/allocine.yaml [--apply]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import get_length_grouped_indices

from tvt.config import load_config
from tvt.data import fingerprint, load_splits
from tvt.evaluation import accuracy, macro_f1
from tvt.utils import load_json


def grouped_order(texts: pd.Series, tokenizer, max_length: int, batch_size: int, seed: int) -> list[int]:
    lengths = [len(ids) for ids in tokenizer(list(texts), truncation=True, max_length=max_length)["input_ids"]]
    return get_length_grouped_indices(lengths, batch_size, generator=torch.Generator().manual_seed(seed))


def recover(pred_path: Path, order: list[int], expected_labels: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    saved = pd.read_csv(pred_path, index_col="test_idx")
    assert np.array_equal(saved["label"].to_numpy(), expected_labels), "label column does not match the test split"
    prob_cols = [c for c in saved.columns if c.startswith("prob_")]
    probs = np.empty((len(saved), len(prob_cols)))
    probs[np.asarray(order)] = saved[prob_cols].to_numpy()  # saved row k is dataset row order[k]
    fixed = pd.DataFrame(probs, columns=prob_cols, index=saved.index)
    fixed.insert(0, "pred", probs.argmax(axis=1))
    fixed.insert(0, "label", expected_labels)
    before = float((saved["pred"] == saved["label"]).mean())
    after = float((fixed["pred"] == fixed["label"]).mean())
    return fixed, before, after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--apply", action="store_true", help="overwrite predictions.csv (a backup is kept)")
    parser.add_argument("--min-accuracy", type=float, default=0.85, help="refuse to write below this accuracy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_splits(cfg.data, cfg.seed)
    expected_fp = load_json(cfg.run_dir / "data_info.json")["test_fingerprint"]
    assert fingerprint(splits.test) == expected_fp, "local test split differs from the one used on Colab"
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer.model_name)
    y = splits.test["label"].to_numpy()
    # Every run predicted the same test set, so the shuffled order only depends on the run's seed.
    tcfg = cfg.transformer

    runs = [(d, int(d.name.split("_")[1])) for d in sorted((cfg.run_dir / "transformer").glob("seed_*"))]
    runs += [(d, int(d.name.split("_s")[1])) for d in sorted((cfg.run_dir / "learning_curve" / "transformer").glob("n*_s*"))]
    orders: dict[int, list[int]] = {}
    rows = []
    for run_dir, seed in runs:
        if seed not in orders:
            orders[seed] = grouped_order(splits.test["text"], tokenizer, tcfg.max_length, tcfg.eval_batch_size, seed)
        backup = run_dir / "predictions_unaligned.csv"
        # Always start from the original (unaligned) file, so the script can be rerun safely.
        fixed, before, after = recover(backup if backup.exists() else run_dir / "predictions.csv", orders[seed], y)
        rows.append({"run": str(run_dir.relative_to(cfg.run_dir)), "acc_before": round(before, 4), "acc_after": round(after, 4)})
        if args.apply:
            if after < args.min_accuracy and "learning_curve" not in str(run_dir):
                raise SystemExit(f"{run_dir}: recovered accuracy {after:.4f} too low, nothing written")
            if not backup.exists():
                (run_dir / "predictions.csv").rename(backup)
            fixed.to_csv(run_dir / "predictions.csv")
    print(pd.DataFrame(rows).to_string(index=False))

    lc_path = cfg.run_dir / "learning_curve" / "learning_curve.csv"
    if args.apply and lc_path.exists():
        # The learning-curve CSV stores scores computed from the unaligned files: recompute them.
        lc = pd.read_csv(lc_path)
        for i, row in lc[lc["model"] == "transformer"].iterrows():
            frame = pd.read_csv(cfg.run_dir / "learning_curve" / "transformer" / f"n{row.train_size}_s{row.seed}" / "predictions.csv")
            lc.loc[i, "test_macro_f1"] = macro_f1(frame["label"].to_numpy(), frame["pred"].to_numpy())
            lc.loc[i, "test_accuracy"] = accuracy(frame["label"].to_numpy(), frame["pred"].to_numpy())
        lc.to_csv(lc_path, index=False)
        print(f"Recomputed transformer rows of {lc_path}")


if __name__ == "__main__":
    main()
