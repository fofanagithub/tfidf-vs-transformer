"""Small shared helpers: seeding, logging, JSON I/O, timing, environment capture."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def get_logger(name: str = "tvt") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_serializable)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Usage: `with timer() as t: ...` then read `t["seconds"]`."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def path_size_mb(path: str | Path) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import sklearn
        import torch
        import transformers

        info.update(
            torch=torch.__version__,
            transformers=transformers.__version__,
            sklearn=sklearn.__version__,
            cuda=torch.cuda.is_available(),
            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
        )
    except ImportError:
        pass
    return info
