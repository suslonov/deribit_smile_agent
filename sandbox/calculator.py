"""Sandboxed feature and signal calculator.

THIS IS THE ONLY FILE THE LLM AGENT MAY REWRITE.

API contract (must not change):
    compute(options_df, config, seed) -> pd.DataFrame

The returned DataFrame must have these columns:
    signal_ts     pd.Timestamp  when the signal is actionable
    asset         str           BTC | ETH
    strategy      str           straddle | vertical_spread
    direction     str           long | short
    expiry        pd.Timestamp  option expiry
    strike        float | NaN   for straddle
    strike_long   float | NaN   for vertical_spread
    strike_short  float | NaN   for vertical_spread
    option_type   str | NaN     C | P  for vertical_spread

All other columns are optional metadata.
No filesystem access, no network access, no randomness beyond the provided seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute(
    options_df: pd.DataFrame,
    config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute features and generate signals from the normalized options DataFrame.

    Parameters
    ----------
    options_df:  Normalized options table (read-only copy).
    config:      Calculator parameters from YAML config['calculator'].
    seed:        Fixed seed for any stochastic components.

    Returns
    -------
    signals_df:  DataFrame with one row per signal (see module docstring).
    """
    cfg = _CalculatorConfig(config)

    assets = cfg.assets
    df = options_df[options_df["asset"].isin(assets)].copy()

    if df.empty:
        return _empty_signals()

    # -----------------------------------------------------------------------
    # Step 1: Time downsampling — one snapshot per hourly bucket per instrument
    # This dramatically reduces the number of rows to process.
    # -----------------------------------------------------------------------
    df["_ts_bucket"] = df["timestamp"].dt.floor(cfg.time_bucket)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["asset", "_ts_bucket", "instrument_name"], keep="first")

    # -----------------------------------------------------------------------
    # Step 2: DTE filter
    # -----------------------------------------------------------------------
    df["_dte"] = (df["expiry_ts"] - df["_ts_bucket"]).dt.total_seconds() / 86400.0
    df = df[(df["_dte"] >= cfg.target_dte_min) & (df["_dte"] <= cfg.target_dte_max)]

    if df.empty:
        return _empty_signals()

    # -----------------------------------------------------------------------
    # Step 3: Compute smile features using fully vectorized aggregations
    # -----------------------------------------------------------------------
    smile_df = _compute_smile_features_vectorized(df, cfg)

    if smile_df.empty:
        return _empty_signals()

    # -----------------------------------------------------------------------
    # Step 4: Straddle signals
    # -----------------------------------------------------------------------
    straddle_df = _build_straddle_signals(df, smile_df, cfg)

    # -----------------------------------------------------------------------
    # Step 5: Vertical spread signals
    # -----------------------------------------------------------------------
    spread_df = _build_spread_signals(df, smile_df, cfg)

    pieces = [p for p in [straddle_df, spread_df] if p is not None and not p.empty]
    if not pieces:
        return _empty_signals()

    result = pd.concat(pieces, ignore_index=True)
    return result.reindex(columns=_OUTPUT_COLS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OUTPUT_COLS = [
    "signal_ts", "asset", "strategy", "direction", "expiry",
    "strike", "strike_long", "strike_short", "option_type",
]

_GROUP_COLS = ["asset", "_ts_bucket", "expiry_ts"]


class _CalculatorConfig:
    def __init__(self, cfg: dict) -> None:
        self.assets: list[str] = cfg.get("assets", ["BTC", "ETH"])
        self.min_oi: float = float(cfg.get("min_oi", 10.0))
        self.min_strikes: int = int(cfg.get("min_strikes", 3))
        self.target_dte_min: float = float(cfg.get("target_dte_min", 7))
        self.target_dte_max: float = float(cfg.get("target_dte_max", 45))
        self.straddle_moneyness_tol: float = float(cfg.get("straddle_moneyness_tol", 0.02))
        self.iv_rv_threshold: float = float(cfg.get("iv_rv_threshold", 0.05))
        self.smile_slope_threshold: float = float(cfg.get("smile_slope_threshold", 0.0))
        self.time_bucket: str = cfg.get("time_bucket", "1h")


def _compute_smile_features_vectorized(
    df: pd.DataFrame,
    cfg: "_CalculatorConfig",
) -> pd.DataFrame:
    """
    Vectorized computation of smile features:
    - ATM IV: mean mark_iv for strikes within 5% of underlying
    - Smile slope: OLS slope of mark_iv ~ log(strike/underlying)

    Uses only pandas groupby().agg() — no Python-level apply loops.
    """
    iv_df = df.dropna(subset=["mark_iv"]).copy()
    iv_df = iv_df[(iv_df["mark_iv"] > 0) & (iv_df["open_interest"] >= cfg.min_oi)]

    if iv_df.empty:
        return pd.DataFrame()

    iv_df["_moneyness"] = iv_df["strike"] / iv_df["underlying_price"]
    iv_df["_log_m"] = np.log(iv_df["_moneyness"])

    # Strike count per group — filter groups below min_strikes
    n_strikes = iv_df.groupby(_GROUP_COLS, observed=True)["strike"].nunique()
    valid_groups = n_strikes[n_strikes >= cfg.min_strikes].reset_index()
    valid_groups = valid_groups.rename(columns={"strike": "_n_strikes"})

    iv_df = iv_df.merge(valid_groups[_GROUP_COLS], on=_GROUP_COLS, how="inner")
    if iv_df.empty:
        return pd.DataFrame()

    # ATM IV
    atm_mask = np.abs(iv_df["_moneyness"] - 1.0) <= 0.05
    atm_df = iv_df[atm_mask]
    atm_iv = (
        atm_df.groupby(_GROUP_COLS, observed=True)["mark_iv"].mean()
        .rename("atm_iv")
        .reset_index()
    )

    # Smile slope using fully vectorized OLS (no apply):
    # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    # where x = log_moneyness, y = mark_iv
    iv_df["_xy"] = iv_df["_log_m"] * iv_df["mark_iv"]
    iv_df["_x2"] = iv_df["_log_m"] ** 2

    grp = iv_df.groupby(_GROUP_COLS, observed=True)
    und_price = grp["underlying_price"].first().rename("underlying_price")
    agg = grp.agg(
        _sum_x=("_log_m", "sum"),
        _sum_y=("mark_iv", "sum"),
        _sum_xy=("_xy", "sum"),
        _sum_x2=("_x2", "sum"),
        _n=("_log_m", "count"),
    ).reset_index()

    numer = agg["_n"] * agg["_sum_xy"] - agg["_sum_x"] * agg["_sum_y"]
    denom = agg["_n"] * agg["_sum_x2"] - agg["_sum_x"] ** 2

    slope_df = agg[_GROUP_COLS].copy()
    slope_df["smile_slope"] = np.where(np.abs(denom) > 1e-10, numer / denom, np.nan)
    slope_df = slope_df.merge(und_price.reset_index(), on=_GROUP_COLS, how="left")

    # Merge all features
    result = atm_iv.merge(
        slope_df[_GROUP_COLS + ["smile_slope", "underlying_price"]],
        on=_GROUP_COLS, how="outer",
    )

    return result


def _build_straddle_signals(
    df: pd.DataFrame,
    smile_df: pd.DataFrame,
    cfg: "_CalculatorConfig",
) -> pd.DataFrame | None:
    """Build straddle signals via vectorized merge operations."""
    feats = smile_df.dropna(subset=["atm_iv"]).copy()
    if feats.empty:
        return None

    feats["direction"] = np.where(feats["atm_iv"] > cfg.iv_rv_threshold, "short", "long")

    # Find ATM strikes with both call and put
    iv_df = df.dropna(subset=["mark_iv"]).copy()
    iv_df = iv_df[(iv_df["mark_iv"] > 0) & (iv_df["open_interest"] >= cfg.min_oi)]
    iv_df["_moneyness"] = iv_df["strike"] / iv_df["underlying_price"]
    atm_iv = iv_df[np.abs(iv_df["_moneyness"] - 1.0) <= cfg.straddle_moneyness_tol]

    if atm_iv.empty:
        return None

    # Check which strikes have both call and put
    calls = (
        atm_iv[atm_iv["option_type"] == "C"]
        .drop_duplicates(subset=_GROUP_COLS + ["strike"])
        [_GROUP_COLS + ["strike"]]
        .assign(_has_call=True)
    )
    puts = (
        atm_iv[atm_iv["option_type"] == "P"]
        .drop_duplicates(subset=_GROUP_COLS + ["strike"])
        [_GROUP_COLS + ["strike"]]
        .assign(_has_put=True)
    )
    both = calls.merge(puts, on=_GROUP_COLS + ["strike"], how="inner")

    if both.empty:
        return None

    # Closest-to-ATM strike per group
    both = both.merge(
        atm_iv[_GROUP_COLS + ["underlying_price"]].drop_duplicates(_GROUP_COLS),
        on=_GROUP_COLS, how="left",
    )
    both["_abs_m"] = np.abs(both["strike"] / both["underlying_price"] - 1.0)
    best = (
        both.sort_values("_abs_m")
        .drop_duplicates(subset=_GROUP_COLS)
        [_GROUP_COLS + ["strike"]]
    )

    signals = feats.merge(best, on=_GROUP_COLS, how="inner")
    if signals.empty:
        return None

    signals = signals.rename(columns={"_ts_bucket": "signal_ts", "expiry_ts": "expiry"})
    signals["strategy"] = "straddle"
    signals["strike_long"] = np.nan
    signals["strike_short"] = np.nan
    signals["option_type"] = np.nan
    return signals


def _build_spread_signals(
    df: pd.DataFrame,
    smile_df: pd.DataFrame,
    cfg: "_CalculatorConfig",
) -> pd.DataFrame | None:
    """Build vertical spread signals based on smile slope."""
    feats = smile_df.dropna(subset=["smile_slope"]).copy()
    feats = feats[np.abs(feats["smile_slope"]) > cfg.smile_slope_threshold]
    if feats.empty:
        return None

    iv_df = df.dropna(subset=["mark_iv"]).copy()
    iv_df = iv_df[(iv_df["mark_iv"] > 0) & (iv_df["open_interest"] >= cfg.min_oi)]
    iv_df["_moneyness"] = iv_df["strike"] / iv_df["underlying_price"]

    rows: list[dict] = []

    # Split by positive/negative slope to reduce iterations
    pos_feats = feats[feats["smile_slope"] > cfg.smile_slope_threshold]
    neg_feats = feats[feats["smile_slope"] < -cfg.smile_slope_threshold]

    def _spread_for_group(
        feat_row: pd.Series,
        opt_type: str,
        moneyness_low: float,
        moneyness_high: float,
        direction: str,
        ascending: bool,
    ) -> dict | None:
        snap = iv_df[
            (iv_df["asset"] == feat_row["asset"])
            & (iv_df["_ts_bucket"] == feat_row["_ts_bucket"])
            & (iv_df["expiry_ts"] == feat_row["expiry_ts"])
            & (iv_df["option_type"] == opt_type)
            & (iv_df["_moneyness"] >= moneyness_low)
            & (iv_df["_moneyness"] <= moneyness_high)
        ].sort_values("strike", ascending=ascending)
        if len(snap) < 2:
            return None
        strike_long = float(snap.iloc[0]["strike"])
        strike_short = float(snap.iloc[1]["strike"])
        return {
            "signal_ts": feat_row["_ts_bucket"],
            "asset": feat_row["asset"],
            "strategy": "vertical_spread",
            "direction": direction,
            "expiry": feat_row["expiry_ts"],
            "strike": np.nan,
            "strike_long": strike_long,
            "strike_short": strike_short,
            "option_type": opt_type,
        }

    for _, row in pos_feats.iterrows():
        entry = _spread_for_group(row, "C", 0.98, 1.10, "short", True)
        if entry:
            rows.append(entry)

    for _, row in neg_feats.iterrows():
        entry = _spread_for_group(row, "P", 0.90, 1.02, "long", False)
        if entry:
            rows.append(entry)

    return pd.DataFrame(rows) if rows else None


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLS)
