"""Transformer fine-tuning with the Hugging Face Trainer.

Protocol: the model is fine-tuned on the train split, the best epoch is selected on validation macro-F1
(with early stopping), and the test split is predicted once. Several seeds are run because fine-tuning
variance is often of the same order as the gap between models.
"""

from __future__ import annotations

import inspect
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .baseline import save_predictions
from .config import TransformerConfig
from .data import Splits
from .evaluation import accuracy, macro_f1
from .utils import get_logger, save_json, set_seed, timer

log = get_logger()


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": macro_f1(labels, preds), "accuracy": accuracy(labels, preds)}


def _version_kwargs(warmup_ratio: float) -> dict:
    """TrainingArguments renamed between transformers 4.x and 5.x (Colab may ship either)."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    # 5.x merged `warmup_ratio` into `warmup_steps` (a float in [0, 1) is read as a ratio).
    kwargs: dict = {"warmup_ratio": warmup_ratio} if "warmup_ratio" in params else {"warmup_steps": warmup_ratio}
    # Batches of similar lengths: much less padding with dynamic padding on long-tailed review lengths.
    if "group_by_length" in params:
        kwargs["group_by_length"] = True
    else:
        kwargs["train_sampling_strategy"] = "group_by_length"
    return kwargs


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    """Class probabilities, row i for texts[i].

    Deliberately not `Trainer.predict`: with transformers 5.x its sampler follows `train_sampling_strategy`,
    so a length-grouped strategy silently returns rows out of order. Here batches are built from
    length-sorted texts (little padding) and the result is scattered back to the input order explicitly.
    """
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    order = np.argsort([len(ids) for ids in enc["input_ids"]], kind="stable")
    collator = DataCollatorWithPadding(tokenizer)
    probs = np.empty((len(texts), model.config.num_labels))
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        batch = collator([{key: enc[key][i] for key in enc.keys()} for i in idx])
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def _tokenize(frame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(frame[["text", "label"]].rename(columns={"label": "labels"}), preserve_index=False)
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def run_transformer_seed(splits: Splits, cfg: TransformerConfig, seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    cuda = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=splits.num_labels,
        id2label=dict(enumerate(splits.label_names)),
        label2id={name: i for i, name in enumerate(splits.label_names)},
    )
    train_ds = _tokenize(splits.train, tokenizer, cfg.max_length)
    val_ds = _tokenize(splits.val, tokenizer, cfg.max_length)
    truncated = float(np.mean([len(ids) >= cfg.max_length for ids in train_ds["input_ids"]]))
    log.info("transformer | seed=%d | %.1f%% of train texts truncated at %d tokens", seed, 100 * truncated, cfg.max_length)

    ckpt_dir = out_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        **_version_kwargs(cfg.warmup_ratio),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=cuda,
        seed=seed,
        data_seed=seed,
        report_to="none",
        dataloader_num_workers=2 if cuda else 0,
        dataloader_pin_memory=cuda,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    with timer() as t_fit:
        trainer.train()
    best_val = trainer.evaluate()

    # Inference timing includes tokenisation, as in production.
    with timer() as t_pred:
        probs = predict_proba(trainer.model, tokenizer, splits.test["text"].tolist(), cfg.max_length, cfg.eval_batch_size)
    save_predictions(out_dir / "predictions.csv", splits.test["label"].to_numpy(), probs)
    save_json(trainer.state.log_history, out_dir / "log_history.json")

    if cfg.save_model:
        trainer.save_model(str(out_dir / "model"))
        tokenizer.save_pretrained(str(out_dir / "model"))
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "model": cfg.model_name,
        "seed": seed,
        "val_macro_f1": float(best_val["eval_macro_f1"]),
        "best_checkpoint_step": trainer.state.best_model_checkpoint and int(trainer.state.best_model_checkpoint.rsplit("-", 1)[-1]),
        "epochs_run": float(trainer.state.epoch),
        "train_truncation_rate": truncated,
        "train_seconds": t_fit["seconds"],
        "inference_seconds": t_pred["seconds"],
        "inference_examples_per_second": len(splits.test) / t_pred["seconds"],
        "inference_device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "n_parameters": int(n_params),
        "model_size_mb": n_params * 4 / 1e6,  # fp32 weights
    }
    save_json(meta, out_dir / "meta.json")
    del trainer, model
    if cuda:
        torch.cuda.empty_cache()
    return meta


def run_transformer(splits: Splits, cfg: TransformerConfig, out_dir: Path, seeds: list[int] | None = None) -> list[dict]:
    metas = []
    for i, seed in enumerate(seeds or cfg.seeds):
        seed_cfg = cfg if i == 0 else replace(cfg, save_model=False)  # keep at most one model on disk
        metas.append(run_transformer_seed(splits, seed_cfg, seed, out_dir / f"seed_{seed}"))
        log.info("transformer | seed=%d done: val macro-F1=%.4f", seed, metas[-1]["val_macro_f1"])
    return metas
