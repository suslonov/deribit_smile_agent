"""Artifact logging: build run bundles and maintain a persistent ledger.

Each run produces a JSON artifact bundle containing:
- run metadata (timestamp, hashes, config)
- summary metrics
- per-fold metrics
- per-strategy / per-horizon breakdowns
- NaN close rate
- top/worst trades
- runtime errors
- data coverage summary
- current sandbox source code

The ledger is a JSONL file (one JSON object per line) for easy streaming reads.
"""
from __future__ import annotations

import json
import logging
import math
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.clock import utc_now_str
from app.utils.hashing import hash_dict, hash_file, short_hash
from app.utils.io import save_json

logger = logging.getLogger(__name__)


def build_artifact(
    run_id: str,
    config: dict,
    config_hash: str,
    sandbox_path: str | Path,
    metrics: dict,
    fold_metrics: list[dict],
    results_df: pd.DataFrame,
    options_df: pd.DataFrame,
    errors: list[str],
    stage: str,
) -> dict:
    """
    Assemble the full artifact bundle for one run.

    Parameters
    ----------
    run_id:        Unique run identifier (timestamp-based).
    config:        Full resolved config dict.
    config_hash:   SHA-256 of the config.
    sandbox_path:  Path to sandbox/calculator.py.
    metrics:       Top-level metrics dict from metrics.compute_metrics().
    fold_metrics:  List of per-fold metric dicts.
    results_df:    Full simulation results DataFrame.
    options_df:    Input options DataFrame (for data coverage summary).
    errors:        List of error strings encountered.
    stage:         'quick' | 'train' | 'test'.
    """
    sandbox_path = Path(sandbox_path)
    sandbox_hash = hash_file(sandbox_path) if sandbox_path.exists() else "missing"
    sandbox_source = sandbox_path.read_text() if sandbox_path.exists() else ""

    return {
        "run_id": run_id,
        "run_ts": utc_now_str(),
        "stage": stage,
        "config_hash": short_hash(config_hash),
        "sandbox_hash": short_hash(sandbox_hash),
        "config": config,
        "sandbox_source": sandbox_source,
        "summary_metrics": _sanitise(metrics.get("summary", {})),
        "by_asset": _sanitise(metrics.get("by_asset", {})),
        "by_strategy": _sanitise(metrics.get("by_strategy", {})),
        "by_horizon": _sanitise(metrics.get("by_horizon", {})),
        "nan_breakdown": metrics.get("nan_breakdown", {}),
        "fold_metrics": [_sanitise(f) for f in fold_metrics],
        "top_trades": _top_trades(results_df, n=10, ascending=False),
        "worst_trades": _top_trades(results_df, n=10, ascending=True),
        "data_coverage": _data_coverage(options_df),
        "errors": errors,
    }


def save_artifact(artifact: dict, artifacts_dir: str | Path) -> Path:
    """Write artifact bundle to disk and append a one-line entry to the ledger."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_id = artifact["run_id"]
    out_path = artifacts_dir / f"{run_id}.json"
    save_json(artifact, out_path)
    logger.info("Artifact saved: %s", out_path)

    ledger_path = artifacts_dir / "ledger.jsonl"
    summary_entry = {
        "run_id": run_id,
        "run_ts": artifact["run_ts"],
        "stage": artifact["stage"],
        "config_hash": artifact["config_hash"],
        "sandbox_hash": artifact["sandbox_hash"],
        "summary_metrics": artifact["summary_metrics"],
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(summary_entry, default=str) + "\n")

    return out_path


def load_ledger(artifacts_dir: str | Path) -> list[dict]:
    ledger_path = Path(artifacts_dir) / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    entries = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _top_trades(
    results_df: pd.DataFrame,
    n: int,
    ascending: bool,
) -> list[dict]:
    if results_df.empty or "pnl" not in results_df.columns:
        return []
    valid = results_df.loc[~results_df["is_nan"]]
    if valid.empty:
        return []
    valid = valid.nsmallest(n, "pnl") if ascending else valid.nlargest(n, "pnl")
    rows = []
    for _, row in valid.iterrows():
        rows.append({
            "signal_ts": str(row.get("signal_ts")),
            "asset": row.get("asset"),
            "strategy": row.get("strategy"),
            "direction": row.get("direction"),
            "horizon_days": row.get("horizon_days"),
            "pnl": _maybe_nan(row.get("pnl")),
            "fees": _maybe_nan(row.get("fees")),
        })
    return rows


def _data_coverage(options_df: pd.DataFrame) -> dict:
    if options_df.empty:
        return {}
    ts = options_df["timestamp"]
    return {
        "start": str(ts.min()),
        "end": str(ts.max()),
        "total_rows": len(options_df),
        "assets": sorted(options_df["asset"].unique().tolist()),
        "unique_timestamps": int(options_df["timestamp"].nunique()),
        "unique_instruments": int(options_df["instrument_name"].nunique()),
        "mark_iv_coverage_pct": float(
            options_df["mark_iv"].notna().mean() * 100
        ),
    }


def _sanitise(obj: Any) -> Any:
    """Replace NaN/inf floats with None for JSON serialisability."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


def _maybe_nan(val: Any) -> Any:
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val
