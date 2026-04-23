"""Compute performance metrics from simulation results.

All metrics operate on the trade-level results DataFrame from simulator.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_metrics(
    results: pd.DataFrame,
    annualise_factor: float = 252.0,
) -> dict:
    """
    Compute top-level and breakdown metrics from simulation results.

    Parameters
    ----------
    results:           Output of simulator.run_simulation().
    annualise_factor:  Trading days per year for Sharpe calculation.

    Returns
    -------
    dict with keys: summary, by_asset, by_strategy, by_horizon, by_expiry_bucket,
                    by_moneyness_bucket.
    """
    if results.empty:
        return {"summary": _empty_summary(), "breakdowns": {}}

    valid = results[~results["is_nan"]].copy()
    return {
        "summary": _summary_metrics(valid, results, annualise_factor),
        "by_asset": _breakdown(valid, results, "asset", annualise_factor),
        "by_strategy": _breakdown(valid, results, "strategy", annualise_factor),
        "by_horizon": _breakdown(valid, results, "horizon_days", annualise_factor),
        "by_expiry_bucket": _breakdown_expiry(valid, results, annualise_factor),
        "by_moneyness_bucket": _breakdown_moneyness(valid, results, annualise_factor),
        "nan_breakdown": _nan_breakdown(results),
    }


def _summary_metrics(
    valid: pd.DataFrame,
    all_results: pd.DataFrame,
    ann: float,
) -> dict:
    pnl = valid["pnl"]
    n_total = len(all_results)
    n_valid = len(valid)
    n_nan = n_total - n_valid
    nan_rate = n_nan / n_total if n_total > 0 else float("nan")

    if n_valid == 0:
        m = _empty_summary()
        m["total_trades"] = n_total
        m["nan_count"] = n_nan
        m["nan_rate"] = nan_rate
        return m

    hit_rate = (pnl > 0).mean()
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    median_pnl = pnl.median()
    std_pnl = pnl.std(ddof=1)
    sharpe = (avg_pnl / std_pnl * math.sqrt(ann)) if std_pnl > 0 else float("nan")
    drawdown = _max_drawdown(pnl)

    return {
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "max_drawdown": drawdown,
        "total_trades": n_total,
        "valid_trades": n_valid,
        "nan_count": n_nan,
        "nan_rate": nan_rate,
    }


def _breakdown(
    valid: pd.DataFrame,
    all_results: pd.DataFrame,
    group_col: str,
    ann: float,
) -> dict:
    out = {}
    for key, grp in valid.groupby(group_col):
        all_grp = all_results[all_results[group_col] == key]
        out[str(key)] = _summary_metrics(grp, all_grp, ann)
    return out


def _breakdown_expiry(
    valid: pd.DataFrame,
    all_results: pd.DataFrame,
    ann: float,
) -> dict:
    """Group by days-to-expiry bucket: <7d, 7-30d, 30-90d, >90d."""
    if valid.empty:
        return {}

    def bucket(row: pd.Series) -> str:
        dte = (row["expiry"] - row["signal_ts"]).days
        if dte < 7:
            return "<7d"
        if dte < 30:
            return "7-30d"
        if dte < 90:
            return "30-90d"
        return ">90d"

    valid = valid.copy()
    valid["_expiry_bucket"] = valid.apply(bucket, axis=1)
    all_results = all_results.copy()
    all_results["_expiry_bucket"] = all_results.apply(
        lambda r: bucket(r) if not r["is_nan"] else "unknown", axis=1
    )
    return _breakdown(valid, all_results, "_expiry_bucket", ann)


def _breakdown_moneyness(
    valid: pd.DataFrame,
    all_results: pd.DataFrame,
    ann: float,
) -> dict:
    """Group by moneyness if 'meta' contains a strike field."""
    if valid.empty:
        return {}

    def moneyness_bucket(row: pd.Series) -> str:
        meta = row.get("meta", {})
        if not isinstance(meta, dict):
            return "unknown"
        strike = meta.get("strike") or meta.get("strike_long")
        if strike is None:
            return "unknown"
        und = row.get("underlying_price_at_signal", float("nan"))
        if math.isnan(float(und)) if und is not None else True:
            return "unknown"
        ratio = float(strike) / float(und)
        if ratio < 0.9:
            return "OTM_low"
        if ratio < 0.97:
            return "slightly_OTM_low"
        if ratio <= 1.03:
            return "ATM"
        if ratio <= 1.1:
            return "slightly_OTM_high"
        return "OTM_high"

    # moneyness requires underlying_price_at_signal column (optional)
    if "underlying_price_at_signal" not in valid.columns:
        return {}

    valid = valid.copy()
    valid["_money_bucket"] = valid.apply(moneyness_bucket, axis=1)
    all_results = all_results.copy()
    all_results["_money_bucket"] = all_results.apply(
        lambda r: moneyness_bucket(r) if not r["is_nan"] else "unknown", axis=1
    )
    return _breakdown(valid, all_results, "_money_bucket", ann)


def _nan_breakdown(results: pd.DataFrame) -> dict:
    nan_rows = results[results["is_nan"]]
    if nan_rows.empty:
        return {}
    return nan_rows["nan_reason"].value_counts().to_dict()


def _max_drawdown(pnl_series: pd.Series) -> float:
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdowns = cumulative - running_max
    return float(drawdowns.min())


def _empty_summary() -> dict:
    return {
        "total_pnl": float("nan"),
        "avg_pnl": float("nan"),
        "median_pnl": float("nan"),
        "std_pnl": float("nan"),
        "sharpe": float("nan"),
        "hit_rate": float("nan"),
        "max_drawdown": float("nan"),
        "total_trades": 0,
        "valid_trades": 0,
        "nan_count": 0,
        "nan_rate": float("nan"),
    }
