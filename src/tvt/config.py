"""Experiment configuration: typed dataclasses loaded from YAML, with dotted CLI overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str  # Hugging Face Hub dataset id
    text_column: str = "text"
    label_column: str = "label"
    language: str = "en"  # used by the error analysis (negation lexicon)
    val_size: float = 0.1  # only used when the dataset has no validation split
    max_train_samples: int | None = None  # stratified subsample, for budget-limited runs
    max_eval_samples: int | None = None  # caps val and test (stratified)


@dataclass
class BaselineConfig:
    # Each feature set is evaluated with every C value; selection uses validation macro-F1.
    feature_sets: list[str] = field(default_factory=lambda: ["word_1_1", "word_1_2", "word_1_2+char_2_5"])
    C_grid: list[float] = field(default_factory=lambda: [0.25, 1.0, 4.0, 16.0])
    min_df: int = 2
    max_features: int | None = 500_000
    class_weight: str | None = None  # "balanced" for imbalanced datasets
    max_iter: int = 2000


@dataclass
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 1
    seeds: list[int] = field(default_factory=lambda: [13, 42, 2024])
    save_model: bool = False  # keep the best checkpoint of the first seed on disk


@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    seed: int = 42  # seed for data splitting / subsampling (shared by both models)
    output_dir: str = "results"
    learning_curve_sizes: list[int] = field(default_factory=lambda: [250, 500, 1000, 2000, 5000, 10000])
    learning_curve_seeds: list[int] = field(default_factory=lambda: [13, 42])
    n_bootstrap: int = 1000

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def _build(cls: type, values: dict[str, Any]):
    """Recursively instantiate a dataclass from a dict, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        sub = _NESTED.get((cls, key))
        kwargs[key] = _build(sub, value or {}) if sub else value
    return cls(**kwargs)


_NESTED = {
    (ExperimentConfig, "data"): DataConfig,
    (ExperimentConfig, "baseline"): BaselineConfig,
    (ExperimentConfig, "transformer"): TransformerConfig,
}


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides; values are parsed as YAML (so 3 -> int, [1,2] -> list, null -> None)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = apply_overrides(raw, overrides or [])
    return _build(ExperimentConfig, raw)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
