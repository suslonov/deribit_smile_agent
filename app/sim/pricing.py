"""Execution model and fee calculation.

Price = execution price for opening/closing a position leg.
Fee   = Deribit options fee: min(0.03% * underlying, 12.5% * option_price)
        per contract, configurable via FeeConfig.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from app.data.schema import ExecutionModel


@dataclass(frozen=True)
class FeeConfig:
    """Per-contract fee model."""
    pct_of_underlying: float   # e.g. 0.0003 for 0.03 %
    max_pct_of_option: float   # e.g. 0.125 for 12.5 %


@dataclass(frozen=True)
class ExecConfig:
    """Execution model configuration."""
    model: ExecutionModel
    slippage_bps: float = 0.0   # only used for CROSSED_BID_ASK_PLUS_SLIPPAGE


def get_exec_price(
    row: pd.Series,
    side: str,        # "buy" or "sell"
    exec_cfg: ExecConfig,
) -> float:
    """
    Return the execution price for one leg, in underlying units.

    Parameters
    ----------
    row:      A row from the normalized options DataFrame.
    side:     'buy' to open a long leg; 'sell' to open a short leg.
    exec_cfg: Execution configuration.
    """
    model = exec_cfg.model

    if model == ExecutionModel.MARK:
        price = _safe(row.get("mark_price"))
        if not math.isnan(price):
            return price

    if model in (ExecutionModel.MID, ExecutionModel.MARK):
        mid = _safe(row.get("mid_price"))
        if not math.isnan(mid):
            return mid
        bid = _safe(row.get("best_bid_price"))
        ask = _safe(row.get("best_ask_price"))
        if not math.isnan(bid) and not math.isnan(ask):
            return (bid + ask) / 2.0
        # fallback to mark
        return _safe(row.get("mark_price"))

    # CROSSED_BID_ASK_PLUS_SLIPPAGE
    bid = _safe(row.get("best_bid_price"))
    ask = _safe(row.get("best_ask_price"))
    slippage = exec_cfg.slippage_bps / 10_000.0

    if side == "buy":
        base = ask if not math.isnan(ask) else _safe(row.get("mid_price"))
        return base * (1.0 + slippage) if not math.isnan(base) else float("nan")
    else:
        base = bid if not math.isnan(bid) else _safe(row.get("mid_price"))
        return base * (1.0 - slippage) if not math.isnan(base) else float("nan")


def compute_fee(
    option_price: float,
    underlying_price: float,
    fee_cfg: FeeConfig,
) -> float:
    """
    Return fee in underlying units for one contract.

    fee = min(pct_of_underlying * underlying, max_pct_of_option * option_price)
    Returns NaN when inputs are NaN.
    """
    if math.isnan(option_price) or math.isnan(underlying_price):
        return float("nan")
    fee = fee_cfg.pct_of_underlying * underlying_price
    cap = fee_cfg.max_pct_of_option * option_price
    return min(fee, cap)


def _safe(val: object) -> float:
    if val is None:
        return float("nan")
    try:
        f = float(val)
        return f if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")
