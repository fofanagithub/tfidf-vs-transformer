"""Command-line entry point.

    tvt baseline        --config configs/allocine.yaml
    tvt transformer     --config configs/allocine.yaml [--set transformer.num_epochs=2]
    tvt learning-curve  --config configs/allocine.yaml
    tvt report          --config configs/allocine.yaml
    tvt all             --config configs/allocine.yaml   # everything, in order
"""

from __future__ import annotations

import argparse

from .config import ExperimentConfig, config_to_dict, load_config
from .data import load_splits
from .utils import environment_info, get_logger, save_json

log = get_logger()


def _baseline(cfg: ExperimentConfig, splits) -> None:
    from .baseline import run_baseline

    meta = run_baseline(splits, cfg.baseline, cfg.run_dir / "baseline")
    log.info("baseline done: val macro-F1=%.4f", meta["val_macro_f1"])


def _transformer(cfg: ExperimentConfig, splits) -> None:
    from .transformer import run_transformer

    # Resumable: seeds whose predictions already exist are skipped.
    todo = [s for s in cfg.transformer.seeds if not (cfg.run_dir / "transformer" / f"seed_{s}" / "meta.json").exists()]
    if len(todo) < len(cfg.transformer.seeds):
        log.info("transformer: skipping finished seeds %s", sorted(set(cfg.transformer.seeds) - set(todo)))
    if todo:
        run_transformer(splits, cfg.transformer, cfg.run_dir / "transformer", seeds=todo)


def _learning_curve(cfg: ExperimentConfig, splits) -> None:
    from .learning_curve import run_learning_curve

    run_learning_curve(splits, cfg)


def _report(cfg: ExperimentConfig, splits) -> None:
    from .report import build_report

    build_report(cfg, splits)


COMMANDS = {"baseline": [_baseline], "transformer": [_transformer], "learning-curve": [_learning_curve], "report": [_report]}
COMMANDS["all"] = [_baseline, _transformer, _learning_curve, _report]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tvt", description="TF-IDF baseline vs fine-tuned Transformer")
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set transformer.num_epochs=2 (repeatable)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": config_to_dict(cfg), "environment": environment_info()}, cfg.run_dir / f"run_{args.command}.json")

    splits = load_splits(cfg.data, cfg.seed)
    save_json(splits.summary(), cfg.run_dir / "data_info.json")
    for step in COMMANDS[args.command]:
        step(cfg, splits)


if __name__ == "__main__":
    main()
