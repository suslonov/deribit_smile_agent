"""Chunked pipeline: runs the calculator day-by-day to keep memory bounded.

Instead of passing the entire date range to the calculator at once, this
module processes the data in configurable `chunk_days` windows and accumulates
signals. The simulator then runs on the full signals + full data.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.data.loader import load_date_range
from app.sim.metrics import compute_metrics
from app.sim.pricing import ExecConfig, FeeConfig
from app.sim.simulator import run_simulation
from sandbox.runner import SandboxRunner

logger = logging.getLogger(__name__)


def run_pipeline(
    data_root: str | Path,
    start: date,
    end: date,
    config: dict,
    sandbox_runner: SandboxRunner,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    horizons: list[int],
    assets: list[str] | None = None,
    cache_dir: str | Path | None = None,
    max_workers: int = 1,
    chunk_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full pipeline (load → features/signals → simulate → metrics)
    in day-sized chunks to keep memory bounded.

    Returns
    -------
    all_signals:  Concatenated signals from all chunks.
    all_results:  Concatenated simulation results for all signals.
    metrics:      Aggregate performance metrics.
    """
    data_root = Path(data_root)
    calc_cfg = config.get("calculator", {})

    # We collect all signals first, then simulate once against the full window.
    # The simulation lookups need a buffer of `max(horizons)` days beyond the
    # last signal date so close prices can be found.
    sim_end = end + timedelta(days=max(horizons) + 1)

    # Process in chunks to compute signals
    all_signals: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)

        chunk_df = load_date_range(
            root=data_root,
            start=chunk_start,
            end=chunk_end,
            assets=assets,
            cache_dir=cache_dir,
            max_workers=max_workers,
        )

        if chunk_df.empty:
            logger.debug("No data for chunk %s – %s", chunk_start, chunk_end)
            chunk_start += timedelta(days=chunk_days)
            continue

        try:
            signals = sandbox_runner.run(chunk_df, calc_cfg)
        except Exception as exc:
            logger.warning("Calculator failed for chunk %s – %s: %s",
                           chunk_start, chunk_end, exc)
            chunk_start += timedelta(days=chunk_days)
            continue

        if not signals.empty:
            all_signals.append(signals)
            logger.info(
                "Chunk %s – %s: %d signals",
                chunk_start, chunk_end, len(signals),
            )

        chunk_start += timedelta(days=chunk_days)

    if not all_signals:
        logger.warning("No signals generated for %s – %s", start, end)
        from app.sim.simulator import _empty_results
        return pd.DataFrame(), _empty_results(), compute_metrics(pd.DataFrame())

    signals_df = pd.concat(all_signals, ignore_index=True)
    logger.info("Total signals: %d", len(signals_df))

    # For simulation we need the price data including close-horizon lookahead.
    # Load just the unique signal dates + close window.
    signal_dates = signals_df["signal_ts"].dt.date.unique()
    sim_start_d = min(signal_dates)
    sim_end_d = sim_end

    logger.info("Loading sim data: %s – %s", sim_start_d, sim_end_d)
    sim_df = load_date_range(
        root=data_root,
        start=sim_start_d,
        end=sim_end_d,
        assets=assets,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )

    results = run_simulation(signals_df, sim_df, exec_cfg, fee_cfg, horizons)
    metrics = compute_metrics(results)

    return signals_df, results, metrics
