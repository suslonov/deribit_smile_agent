"""Position builders: resolve signal → list of (instrument_name, side) legs.

A signal from the calculator specifies what to trade; these helpers find
the actual instrument rows in the options DataFrame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Leg:
    """One option leg."""
    instrument_name: str
    side: str          # "buy" | "sell"
    qty: float = 1.0   # contracts (always positive)


@dataclass
class Position:
    """Multi-leg position derived from one signal row."""
    signal_ts: pd.Timestamp
    asset: str
    strategy: str
    direction: str
    expiry: pd.Timestamp
    legs: list[Leg]
    meta: dict        # extra info (strike etc.) for reporting


def build_straddle(
    signal_row: pd.Series,
    snapshot: pd.DataFrame,
) -> Optional[Position]:
    """
    Build a straddle position from a signal row.

    Looks for ATM call and put with matching (expiry, strike) in *snapshot*.
    """
    asset = signal_row["asset"]
    expiry = signal_row["expiry"]
    strike = signal_row.get("strike")
    direction = signal_row["direction"]

    if strike is None or (isinstance(strike, float) and np.isnan(strike)):
        logger.debug("Straddle signal missing strike for %s", asset)
        return None

    strike = float(strike)

    mask = (
        (snapshot["asset"] == asset)
        & (snapshot["expiry_ts"] == expiry)
        & (snapshot["strike"] == strike)
    )
    legs_df = snapshot[mask]

    call_rows = legs_df[legs_df["option_type"] == "C"]
    put_rows = legs_df[legs_df["option_type"] == "P"]

    if call_rows.empty or put_rows.empty:
        logger.debug(
            "Straddle: missing call or put for %s expiry=%s strike=%.0f",
            asset, expiry, strike,
        )
        return None

    call_name = call_rows.iloc[0]["instrument_name"]
    put_name = put_rows.iloc[0]["instrument_name"]

    if direction == "long":
        legs = [Leg(call_name, "buy"), Leg(put_name, "buy")]
    else:
        legs = [Leg(call_name, "sell"), Leg(put_name, "sell")]

    return Position(
        signal_ts=signal_row["signal_ts"],
        asset=asset,
        strategy="straddle",
        direction=direction,
        expiry=expiry,
        legs=legs,
        meta={"strike": strike},
    )


def build_vertical_spread(
    signal_row: pd.Series,
    snapshot: pd.DataFrame,
) -> Optional[Position]:
    """
    Build a vertical spread position from a signal row.

    signal_row must have:
      - strike_long, strike_short: the two strikes
      - option_type: 'C' or 'P'
      - direction: 'long' (debit spread) or 'short' (credit spread)
    """
    asset = signal_row["asset"]
    expiry = signal_row["expiry"]
    direction = signal_row["direction"]
    opt_type = signal_row.get("option_type")
    strike_long = signal_row.get("strike_long")
    strike_short = signal_row.get("strike_short")

    missing = []
    if opt_type is None or (isinstance(opt_type, float) and np.isnan(opt_type)):
        missing.append("option_type")
    if strike_long is None or (isinstance(strike_long, float) and np.isnan(strike_long)):
        missing.append("strike_long")
    if strike_short is None or (isinstance(strike_short, float) and np.isnan(strike_short)):
        missing.append("strike_short")

    if missing:
        logger.debug("Vertical spread signal missing fields %s", missing)
        return None

    strike_long = float(strike_long)
    strike_short = float(strike_short)

    mask_type = (
        (snapshot["asset"] == asset)
        & (snapshot["expiry_ts"] == expiry)
        & (snapshot["option_type"] == opt_type)
    )
    opt_df = snapshot[mask_type]

    long_rows = opt_df[opt_df["strike"] == strike_long]
    short_rows = opt_df[opt_df["strike"] == strike_short]

    if long_rows.empty or short_rows.empty:
        logger.debug(
            "Vertical: missing leg(s) for %s %s expiry=%s long=%.0f short=%.0f",
            asset, opt_type, expiry, strike_long, strike_short,
        )
        return None

    long_name = long_rows.iloc[0]["instrument_name"]
    short_name = short_rows.iloc[0]["instrument_name"]

    if direction == "long":
        legs = [Leg(long_name, "buy"), Leg(short_name, "sell")]
    else:
        legs = [Leg(long_name, "sell"), Leg(short_name, "buy")]

    return Position(
        signal_ts=signal_row["signal_ts"],
        asset=asset,
        strategy="vertical_spread",
        direction=direction,
        expiry=expiry,
        legs=legs,
        meta={"strike_long": strike_long, "strike_short": strike_short,
              "option_type": opt_type},
    )


STRATEGY_BUILDERS = {
    "straddle": build_straddle,
    "vertical_spread": build_vertical_spread,
}


def build_position(
    signal_row: pd.Series,
    snapshot: pd.DataFrame,
) -> Optional[Position]:
    """Dispatch to the correct builder based on signal strategy field."""
    strategy = signal_row.get("strategy", "")
    builder = STRATEGY_BUILDERS.get(strategy)
    if builder is None:
        logger.warning("Unknown strategy %r in signal", strategy)
        return None
    return builder(signal_row, snapshot)
