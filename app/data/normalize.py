"""Normalize raw Deribit option records into the canonical DataFrame schema."""
from __future__ import annotations

import re
from datetime import timezone

import numpy as np
import pandas as pd

from app.data.schema import OPTIONS_COLUMNS, OPTIONS_DTYPES

_EXPIRY_RE = re.compile(r"^(\d{1,2})([A-Z]{3})(\d{2})$")
_INSTRUMENT_RE = re.compile(r"^(BTC|ETH)-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP])$")
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_expiry(expiry_str: str) -> pd.Timestamp:
    """Parse Deribit expiry string like '31JAN25' → UTC Timestamp at 08:00."""
    m = _EXPIRY_RE.match(expiry_str)
    if not m:
        raise ValueError(f"Cannot parse expiry: {expiry_str!r}")
    day = int(m.group(1))
    month = _MONTH_MAP[m.group(2)]
    year = 2000 + int(m.group(3))
    return pd.Timestamp(year=year, month=month, day=day, hour=8, tz="UTC")


def parse_instrument_name(name: str) -> tuple[str, pd.Timestamp, float, str]:
    """Return (asset, expiry_ts, strike, option_type) for an instrument name."""
    m = _INSTRUMENT_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse instrument: {name!r}")
    asset = m.group(1)
    expiry_ts = parse_expiry(m.group(2))
    strike = float(m.group(3))
    option_type = m.group(4)
    return asset, expiry_ts, strike, option_type


def normalize_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of raw Deribit option dicts (from one snapshot file)
    into a normalized DataFrame with canonical column schema.

    Rows that fail instrument parsing are dropped silently (logged via counter).
    """
    if not records:
        return _empty_frame()

    rows: list[dict] = []
    for rec in records:
        name = rec.get("instrument_name", "")
        try:
            asset, expiry_ts, strike, opt_type = parse_instrument_name(name)
        except ValueError:
            continue

        ts_ms = rec.get("creation_timestamp")
        if ts_ms is None:
            continue
        timestamp = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")

        underlying_price = _safe_float(rec.get("underlying_price"))
        if np.isnan(underlying_price):
            continue

        bid = _safe_float(rec.get("bid_price"))
        ask = _safe_float(rec.get("ask_price"))
        mid = _safe_float(rec.get("mid_price"))
        if np.isnan(mid) and not (np.isnan(bid) or np.isnan(ask)):
            mid = (bid + ask) / 2.0

        mark_iv_raw = rec.get("mark_iv")
        mark_iv = _safe_float(mark_iv_raw)
        # Deribit sometimes returns mark_iv as a percentage (e.g. 75.0 meaning 75%)
        # Normalise to decimal fraction
        if not np.isnan(mark_iv) and mark_iv > 5.0:
            mark_iv = mark_iv / 100.0

        rows.append({
            "timestamp": timestamp,
            "instrument_name": name,
            "asset": asset,
            "expiry_ts": expiry_ts,
            "strike": strike,
            "option_type": opt_type,
            "mark_price": _safe_float(rec.get("mark_price")),
            "mark_iv": mark_iv,
            "underlying_price": underlying_price,
            "best_bid_price": bid,
            "best_ask_price": ask,
            "mid_price": mid,
            "open_interest": _safe_float(rec.get("open_interest")),
            "volume": _safe_float(rec.get("volume")),
        })

    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows, columns=OPTIONS_COLUMNS)
    for col, dtype in OPTIONS_DTYPES.items():
        df[col] = df[col].astype(dtype)
    return df


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OPTIONS_COLUMNS)


def _safe_float(val: object) -> float:
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")
