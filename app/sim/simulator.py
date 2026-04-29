"""Core simulation engine.

For each signal → Position, the simulator:
1. Opens at the first available snapshot at or after signal_ts.
2. Closes at target_ts = signal_ts + horizon_days.
   - Takes the first available snapshot >= target_ts.
   - If the gap from target_ts to the first available row > MAX_GAP_DAYS → NaN.
   - If expiry occurs before close target → return NaN (unless explicitly handled).
3. Records PnL per leg, total PnL, fees, and metadata.

Positions are expressed in underlying units (BTC or ETH).
"""
from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.sim.positions import Leg, Position, build_position
from app.sim.pricing import ExecConfig, FeeConfig, compute_fee, get_exec_price

logger = logging.getLogger(__name__)

MAX_GAP_DAYS = 3   # if close timestamp exceeds target by more than this → NaN


@dataclass
class TradeResult:
    """Result of simulating one position at one holding horizon."""
    position_id: int
    signal_ts: pd.Timestamp
    open_ts: pd.Timestamp
    close_ts: Optional[pd.Timestamp]
    asset: str
    strategy: str
    direction: str
    expiry: pd.Timestamp
    horizon_days: int
    open_pnl: float          # total execution cost at open (negative = paid)
    close_pnl: float         # total proceeds at close
    fees: float              # total fees paid (always positive)
    pnl: float               # net PnL = close_pnl + open_pnl - fees
    is_nan: bool             # True when close could not be determined
    nan_reason: str          # explanation when is_nan
    meta: dict = field(default_factory=dict)


class _TimestampIndex:
    """O(log n) lookup of option snapshots using pandas searchsorted."""

    def __init__(self, options_df: pd.DataFrame) -> None:
        self._df = options_df.sort_values("timestamp").reset_index(drop=True)
        self._ts = pd.to_datetime(self._df["timestamp"], utc=True)

    def get_snapshot_at_or_after(
        self,
        target_ts: pd.Timestamp,
        window_minutes: int = 5,
    ) -> pd.DataFrame:
        """Return rows for the first minute-snapshot >= target_ts."""
        if self._ts.empty:
            return pd.DataFrame()
        target = pd.Timestamp(target_ts)
        if target.tzinfo is None:
            target = target.tz_localize("UTC")
        idx = self._ts.searchsorted(target, side="left")
        if idx >= len(self._ts):
            return pd.DataFrame()
        first_ts = self._ts.iloc[idx]
        snap_end = first_ts + pd.Timedelta(minutes=window_minutes)
        end_idx = self._ts.searchsorted(snap_end, side="right")
        return self._df.iloc[idx:end_idx]


def run_simulation(
    signals_df: pd.DataFrame,
    options_df: pd.DataFrame,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Simulate all signals and return a flat DataFrame of TradeResult rows.

    Parameters
    ----------
    signals_df:  Output of the sandbox calculator (signal rows).
    options_df:  Full normalized options DataFrame for the period.
    exec_cfg:    Execution model config.
    fee_cfg:     Fee config.
    horizons:    List of holding periods in days, e.g. [1, 3].

    Returns
    -------
    DataFrame with one row per (signal × horizon).
    """
    if signals_df.empty:
        logger.warning("No signals to simulate")
        return _empty_results()

    # Build fast binary-search timestamp index
    ts_index = _TimestampIndex(options_df)

    results: list[TradeResult] = []
    for pos_id, sig_row in enumerate(signals_df.itertuples(index=False)):
        sig_series = sig_row._asdict()
        signal_ts = sig_series["signal_ts"]

        # Find the open snapshot: first available at or after signal_ts
        open_snap = ts_index.get_snapshot_at_or_after(signal_ts)
        if open_snap.empty:
            for h in horizons:
                results.append(_nan_result(
                    pos_id, sig_series, signal_ts, h, "no_open_snapshot",
                ))
            continue

        open_ts = open_snap["timestamp"].iloc[0]

        # Build the position (resolve instrument names)
        position = build_position(sig_series, open_snap)
        if position is None:
            for h in horizons:
                results.append(_nan_result(
                    pos_id, sig_series, signal_ts, h, "position_build_failed",
                ))
            continue

        # Compute open cost (paying for long legs, receiving for short)
        open_pnl, open_ok = _compute_leg_pnl(
            position.legs, open_snap, exec_cfg, fee_cfg, side="open"
        )

        for horizon in horizons:
            target_ts = signal_ts + pd.Timedelta(days=horizon)

            # Check expiry before close
            if position.expiry <= target_ts:
                results.append(_nan_result(
                    pos_id, sig_series, signal_ts, horizon,
                    "expiry_before_close",
                ))
                continue

            close_snap = ts_index.get_snapshot_at_or_after(target_ts)
            if close_snap.empty:
                results.append(_nan_result(
                    pos_id, sig_series, signal_ts, horizon, "no_close_snapshot",
                ))
                continue

            close_ts = close_snap["timestamp"].iloc[0]
            gap = (close_ts - target_ts).total_seconds() / 86400.0
            if gap > MAX_GAP_DAYS:
                results.append(_nan_result(
                    pos_id, sig_series, signal_ts, horizon, "close_gap_exceeded",
                ))
                continue

            close_pnl, close_ok = _compute_leg_pnl(
                position.legs, close_snap, exec_cfg, fee_cfg, side="close"
            )

            total_fees = _total_fees(position.legs, open_snap, close_snap, fee_cfg)
            net_pnl = close_pnl + open_pnl - total_fees

            if not open_ok or not close_ok or math.isnan(net_pnl):
                results.append(_nan_result(
                    pos_id, sig_series, signal_ts, horizon, "price_missing",
                ))
                continue

            results.append(TradeResult(
                position_id=pos_id,
                signal_ts=signal_ts,
                open_ts=open_ts,
                close_ts=close_ts,
                asset=position.asset,
                strategy=position.strategy,
                direction=position.direction,
                expiry=position.expiry,
                horizon_days=horizon,
                open_pnl=open_pnl,
                close_pnl=close_pnl,
                fees=total_fees,
                pnl=net_pnl,
                is_nan=False,
                nan_reason="",
                meta=position.meta,
            ))

    if not results:
        return _empty_results()

    rows = [vars(r) for r in results]
    return pd.DataFrame(rows)


def _get_snapshot_at_or_after(
    ts_sorted: pd.DataFrame,
    target_ts: pd.Timestamp,
    window_minutes: int = 5,
) -> pd.DataFrame:
    """
    Return rows from the nearest minute-snapshot at or after *target_ts*.
    Uses a 5-minute window to capture all instruments in that snapshot.
    """
    mask = ts_sorted["timestamp"] >= target_ts
    candidates = ts_sorted[mask]
    if candidates.empty:
        return candidates

    first_ts = candidates["timestamp"].iloc[0]
    snap_end = first_ts + pd.Timedelta(minutes=window_minutes)
    return candidates[candidates["timestamp"] <= snap_end].reset_index(drop=True)


def _compute_leg_pnl(
    legs: list[Leg],
    snapshot: pd.DataFrame,
    exec_cfg: ExecConfig,
    fee_cfg: FeeConfig,
    side: str,          # "open" or "close"
) -> tuple[float, bool]:
    """
    Compute signed PnL contribution from all legs at open or close.

    At open:  long legs pay (−price), short legs receive (+price).
    At close: long legs receive (+price), short legs pay (−price).
    Returns (total_pnl, all_prices_available).
    """
    total = 0.0
    ok = True
    for leg in legs:
        row = _find_instrument(snapshot, leg.instrument_name)
        if row is None:
            ok = False
            total = float("nan")
            continue

        # For open: buying a call = pay ask; closing: sell at bid
        if side == "open":
            exec_side = "buy" if leg.side == "buy" else "sell"
        else:
            # Close reverses the side
            exec_side = "sell" if leg.side == "buy" else "buy"

        price = get_exec_price(row, exec_side, exec_cfg)
        if math.isnan(price):
            ok = False
            total = float("nan")
            continue

        if not math.isnan(total):
            # Open: buy legs = negative contribution; sell legs = positive
            # Close: buy legs = positive; sell legs = negative
            if side == "open":
                total += -price * leg.qty if leg.side == "buy" else price * leg.qty
            else:
                total += price * leg.qty if leg.side == "buy" else -price * leg.qty

    return total, ok


def _total_fees(
    legs: list[Leg],
    open_snap: pd.DataFrame,
    close_snap: pd.DataFrame,
    fee_cfg: FeeConfig,
) -> float:
    """Sum fees for all legs at open and close."""
    total = 0.0
    for leg in legs:
        for snap in (open_snap, close_snap):
            row = _find_instrument(snap, leg.instrument_name)
            if row is None:
                continue
            price = row.get("mark_price", float("nan"))
            if isinstance(price, pd.Series):
                price = price.iloc[0]
            und = row.get("underlying_price", float("nan"))
            if isinstance(und, pd.Series):
                und = und.iloc[0]
            fee = compute_fee(
                float(price) if price is not None else float("nan"),
                float(und) if und is not None else float("nan"),
                fee_cfg,
            )
            if not math.isnan(fee):
                total += fee * leg.qty
    return total


def _find_instrument(
    snapshot: pd.DataFrame,
    instrument_name: str,
) -> Optional[pd.Series]:
    """Find the row for *instrument_name* in a snapshot."""
    rows = snapshot[snapshot["instrument_name"] == instrument_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _nan_result(
    pos_id: int,
    sig_series: dict,
    signal_ts: pd.Timestamp,
    horizon: int,
    reason: str,
) -> TradeResult:
    return TradeResult(
        position_id=pos_id,
        signal_ts=signal_ts,
        open_ts=signal_ts,
        close_ts=None,
        asset=sig_series.get("asset", ""),
        strategy=sig_series.get("strategy", ""),
        direction=sig_series.get("direction", ""),
        expiry=sig_series.get("expiry", pd.NaT),
        horizon_days=horizon,
        open_pnl=float("nan"),
        close_pnl=float("nan"),
        fees=float("nan"),
        pnl=float("nan"),
        is_nan=True,
        nan_reason=reason,
        meta={},
    )


def _empty_results() -> pd.DataFrame:
    cols = [
        "position_id", "signal_ts", "open_ts", "close_ts",
        "asset", "strategy", "direction", "expiry", "horizon_days",
        "open_pnl", "close_pnl", "fees", "pnl", "is_nan", "nan_reason", "meta",
    ]
    return pd.DataFrame(columns=cols)
