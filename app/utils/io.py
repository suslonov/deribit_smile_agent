"""I/O utilities: YAML loading, JSON serialisation, parquet helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
