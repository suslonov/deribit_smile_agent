from __future__ import annotations

import numpy as np
import pandas as pd

_OUTPUT_COLS = [
    "signal_ts", "asset", "strategy", "direction", "expiry",
    "strike", "strike_long", "strike_short", "option_type",
]
_GROUP_COLS = ["asset", "_ts_bucket", "expiry_ts"]


def compute(options_df: pd.DataFrame, config: dict, seed: int = 42) -> pd.DataFrame:
    assets = config.get("assets", ["BTC", "ETH"])
    min_oi = float(config.get("min_oi", 10.0))
    min_strikes = int(config.get("min_strikes", 3))
    dte_min = float(config.get("target_dte_min", 7))
    dte_max = float(config.get("target_dte_max", 45))
    moneyness_tol = float(config.get("straddle_moneyness_tol", 0.02))
    iv_rv_thr = float(config.get("iv_rv_threshold", 0.05))
    slope_thr = float(config.get("smile_slope_threshold", 0.0))
    time_bucket = config.get("time_bucket", "1h")

    empty = pd.DataFrame(columns=_OUTPUT_COLS)

    df = options_df[options_df["asset"].isin(assets)].copy()
    if df.empty:
        return empty

    df["_ts_bucket"] = df["timestamp"].dt.floor(time_bucket)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["asset", "_ts_bucket", "instrument_name"], keep="first")

    df["_dte"] = (df["expiry_ts"] - df["_ts_bucket"]).dt.total_seconds() / 86400.0
    df = df[(df["_dte"] >= dte_min) & (df["_dte"] <= dte_max)]
    if df.empty:
        return empty

    iv_df = df.dropna(subset=["mark_iv"]).copy()
    iv_df = iv_df[(iv_df["mark_iv"] > 0) & (iv_df["open_interest"] >= min_oi)]
    if iv_df.empty:
        return empty

    iv_df["_moneyness"] = iv_df["strike"] / iv_df["underlying_price"]
    iv_df["_log_m"] = np.log(iv_df["_moneyness"])

    n_strikes = iv_df.groupby(_GROUP_COLS, observed=True)["strike"].nunique()
    valid_groups = n_strikes[n_strikes >= min_strikes].reset_index()[_GROUP_COLS]
    iv_df = iv_df.merge(valid_groups, on=_GROUP_COLS, how="inner")
    if iv_df.empty:
        return empty

    atm_iv = (
        iv_df[np.abs(iv_df["_moneyness"] - 1.0) <= 0.05]
        .groupby(_GROUP_COLS, observed=True)["mark_iv"].mean()
        .rename("atm_iv").reset_index()
    )

    iv_df["_xy"] = iv_df["_log_m"] * iv_df["mark_iv"]
    iv_df["_x2"] = iv_df["_log_m"] ** 2
    grp = iv_df.groupby(_GROUP_COLS, observed=True)
    und_price = grp["underlying_price"].first().rename("underlying_price")
    agg = grp.agg(
        _sum_x=("_log_m", "sum"), _sum_y=("mark_iv", "sum"),
        _sum_xy=("_xy", "sum"), _sum_x2=("_x2", "sum"), _n=("_log_m", "count"),
    ).reset_index()
    numer = agg["_n"] * agg["_sum_xy"] - agg["_sum_x"] * agg["_sum_y"]
    denom = agg["_n"] * agg["_sum_x2"] - agg["_sum_x"] ** 2
    slope_df = agg[_GROUP_COLS].copy()
    slope_df["smile_slope"] = np.where(np.abs(denom) > 1e-10, numer / denom, np.nan)
    slope_df = slope_df.merge(und_price.reset_index(), on=_GROUP_COLS, how="left")

    smile_df = atm_iv.merge(
        slope_df[_GROUP_COLS + ["smile_slope", "underlying_price"]],
        on=_GROUP_COLS, how="outer",
    )

    pieces = []

    feats = smile_df.dropna(subset=["atm_iv"]).copy()
    if not feats.empty:
        feats["direction"] = np.where(feats["atm_iv"] > iv_rv_thr, "short", "long")
        atm_rows = iv_df[np.abs(iv_df["_moneyness"] - 1.0) <= moneyness_tol]
        calls = (
            atm_rows[atm_rows["option_type"] == "C"]
            .drop_duplicates(subset=_GROUP_COLS + ["strike"])[_GROUP_COLS + ["strike"]]
            .assign(_has_call=True)
        )
        puts = (
            atm_rows[atm_rows["option_type"] == "P"]
            .drop_duplicates(subset=_GROUP_COLS + ["strike"])[_GROUP_COLS + ["strike"]]
            .assign(_has_put=True)
        )
        both = calls.merge(puts, on=_GROUP_COLS + ["strike"], how="inner")
        if not both.empty:
            both = both.merge(
                atm_rows[_GROUP_COLS + ["underlying_price"]].drop_duplicates(_GROUP_COLS),
                on=_GROUP_COLS, how="left",
            )
            both["_abs_m"] = np.abs(both["strike"] / both["underlying_price"] - 1.0)
            best = (
                both.sort_values("_abs_m")
                .drop_duplicates(subset=_GROUP_COLS)[_GROUP_COLS + ["strike"]]
            )
            signals = feats.merge(best, on=_GROUP_COLS, how="inner")
            if not signals.empty:
                signals = signals.rename(columns={"_ts_bucket": "signal_ts", "expiry_ts": "expiry"})
                signals["strategy"] = "straddle"
                signals["strike_long"] = np.nan
                signals["strike_short"] = np.nan
                signals["option_type"] = np.nan
                pieces.append(signals)

    feats = smile_df.dropna(subset=["smile_slope"]).copy()
    feats = feats[np.abs(feats["smile_slope"]) > slope_thr]
    if not feats.empty:
        rows: list[dict] = []
        for _, row in feats[feats["smile_slope"] > slope_thr].iterrows():
            snap = iv_df[
                (iv_df["asset"] == row["asset"])
                & (iv_df["_ts_bucket"] == row["_ts_bucket"])
                & (iv_df["expiry_ts"] == row["expiry_ts"])
                & (iv_df["option_type"] == "C")
                & (iv_df["_moneyness"] >= 0.98) & (iv_df["_moneyness"] <= 1.10)
            ].sort_values("strike", ascending=True)
            if len(snap) >= 2:
                rows.append({
                    "signal_ts": row["_ts_bucket"], "asset": row["asset"],
                    "strategy": "vertical_spread", "direction": "short",
                    "expiry": row["expiry_ts"], "strike": np.nan,
                    "strike_long": float(snap.iloc[0]["strike"]),
                    "strike_short": float(snap.iloc[1]["strike"]), "option_type": "C",
                })
        for _, row in feats[feats["smile_slope"] < -slope_thr].iterrows():
            snap = iv_df[
                (iv_df["asset"] == row["asset"])
                & (iv_df["_ts_bucket"] == row["_ts_bucket"])
                & (iv_df["expiry_ts"] == row["expiry_ts"])
                & (iv_df["option_type"] == "P")
                & (iv_df["_moneyness"] >= 0.90) & (iv_df["_moneyness"] <= 1.02)
            ].sort_values("strike", ascending=False)
            if len(snap) >= 2:
                rows.append({
                    "signal_ts": row["_ts_bucket"], "asset": row["asset"],
                    "strategy": "vertical_spread", "direction": "long",
                    "expiry": row["expiry_ts"], "strike": np.nan,
                    "strike_long": float(snap.iloc[0]["strike"]),
                    "strike_short": float(snap.iloc[1]["strike"]), "option_type": "P",
                })
        if rows:
            pieces.append(pd.DataFrame(rows))

    if not pieces:
        return empty

    return pd.concat(pieces, ignore_index=True).reindex(columns=_OUTPUT_COLS)
