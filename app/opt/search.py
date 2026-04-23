"""Walk-forward parameter search.

For each fold:
1. Run the sandbox calculator on the fold's training window.
2. Simulate trades.
3. Compute metrics.
4. Optionally grid-search over the calculator's parameter space.
5. Pick the best config by the configured objective.
6. Evaluate on the validation window with the best config.
"""
from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.sim.metrics import compute_metrics
from app.sim.pricing import ExecConfig, FeeConfig
from app.sim.simulator import run_simulation
from app.split.time_split import Split, get_fold_data
from sandbox.runner import SandboxRunner

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    fold: int
    best_config: dict
    train_metrics: dict
    val_metrics: dict


def walk_forward_search(
    folds: list[Split],
    options_df: pd.DataFrame,
    config: dict,
    sandbox_runner: SandboxRunner,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    horizons: list[int],
) -> list[SearchResult]:
    """
    Run walk-forward search across all folds.

    For each fold, optionally grid-search the calculator parameter space
    on the training window, then evaluate the best config on validation.
    """
    search_space = config.get("optimization", {}).get("search_space", {})
    objective = config.get("promotion", {}).get("objective", "sharpe")
    results: list[SearchResult] = []

    for fold in folds:
        logger.info("Processing fold %d: %s – %s", fold.fold, fold.train_start, fold.val_end)
        train_df, val_df = get_fold_data(options_df, fold)

        if train_df.empty or val_df.empty:
            logger.warning("Fold %d: empty train or val set, skipping", fold.fold)
            continue

        # Grid search on training set
        best_cfg, train_metrics = _grid_search(
            train_df, config, sandbox_runner, exec_cfg, fee_cfg, horizons,
            search_space, objective,
        )

        # Evaluate best config on validation set
        val_metrics = _evaluate(val_df, best_cfg, sandbox_runner, exec_cfg, fee_cfg, horizons)

        results.append(SearchResult(
            fold=fold.fold,
            best_config=best_cfg,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        ))
        logger.info(
            "Fold %d done. Train %s=%.4f | Val %s=%.4f",
            fold.fold,
            objective,
            _get_obj(train_metrics, objective),
            objective,
            _get_obj(val_metrics, objective),
        )

    return results


def _grid_search(
    train_df: pd.DataFrame,
    config: dict,
    runner: SandboxRunner,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    horizons: list[int],
    search_space: dict[str, list[Any]],
    objective: str,
) -> tuple[dict, dict]:
    """Return (best_config, best_train_metrics). Falls back to base config if no space."""
    base_calc_cfg = config.get("calculator", {})
    if not search_space:
        metrics = _evaluate(train_df, config, runner, exec_cfg, fee_cfg, horizons)
        return config, metrics

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    best_obj = float("-inf")
    best_cfg = config
    best_metrics: dict = {}

    for combo in itertools.product(*values):
        candidate_calc = copy.deepcopy(base_calc_cfg)
        for k, v in zip(keys, combo):
            candidate_calc[k] = v

        candidate_config = copy.deepcopy(config)
        candidate_config["calculator"] = candidate_calc

        try:
            metrics = _evaluate(
                train_df, candidate_config, runner, exec_cfg, fee_cfg, horizons
            )
        except Exception as exc:
            logger.debug("Grid combo %s failed: %s", combo, exc)
            continue

        obj_val = _get_obj(metrics, objective)
        if obj_val > best_obj:
            best_obj = obj_val
            best_cfg = candidate_config
            best_metrics = metrics

    return best_cfg, best_metrics


def _evaluate(
    df: pd.DataFrame,
    config: dict,
    runner: SandboxRunner,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    horizons: list[int],
) -> dict:
    """Run calc + sim + metrics on *df* using *config*."""
    calc_cfg = config.get("calculator", {})
    signals = runner.run(df, calc_cfg)
    results = run_simulation(signals, df, exec_cfg, fee_cfg, horizons)
    return compute_metrics(results)


def _get_obj(metrics: dict, objective: str) -> float:
    val = metrics.get("summary", {}).get(objective)
    if val is None:
        return float("-inf")
    try:
        f = float(val)
        return f if not (f != f) else float("-inf")  # NaN check
    except (TypeError, ValueError):
        return float("-inf")
