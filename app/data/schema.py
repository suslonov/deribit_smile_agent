"""Normalized data schemas for options and underlying price data."""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator


class Asset(str, Enum):
    BTC = "BTC"
    ETH = "ETH"


class OptionType(str, Enum):
    CALL = "C"
    PUT = "P"


class ExecutionModel(str, Enum):
    MID = "mid"
    MARK = "mark"
    CROSSED_BID_ASK_PLUS_SLIPPAGE = "crossed_bid_ask_plus_slippage"


class Strategy(str, Enum):
    STRADDLE = "straddle"
    VERTICAL_SPREAD = "vertical_spread"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


# Canonical column names for the normalized options DataFrame
OPTIONS_COLUMNS = [
    "timestamp",           # pd.Timestamp UTC
    "instrument_name",     # str  e.g. BTC-31JAN25-90000-C
    "asset",               # str  BTC | ETH
    "expiry_ts",           # pd.Timestamp UTC  (8:00 UTC delivery)
    "strike",              # float
    "option_type",         # str  C | P
    "mark_price",          # float  in underlying units
    "mark_iv",             # float | NaN  annualised decimal (e.g. 0.75 = 75%)
    "underlying_price",    # float  index price
    "best_bid_price",      # float | NaN  in underlying units
    "best_ask_price",      # float | NaN  in underlying units
    "mid_price",           # float | NaN  (bid+ask)/2
    "open_interest",       # float
    "volume",              # float  in contracts
]

OPTIONS_DTYPES: dict[str, str] = {
    "instrument_name": "string",
    "asset": "string",
    "option_type": "string",
}

# Columns required to be non-null for a row to be usable in simulation
REQUIRED_FOR_SIM = ["timestamp", "instrument_name", "asset", "expiry_ts",
                    "strike", "option_type", "underlying_price"]


class RawRecord(BaseModel):
    """Pydantic model matching the raw pickle dict fields."""
    instrument_name: str
    creation_timestamp: int           # ms since epoch
    underlying_price: float
    mark_price: Optional[float] = None
    mark_iv: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    mid_price: Optional[float] = None
    open_interest: Optional[float] = None
    volume: Optional[float] = None
    base_currency: Optional[str] = None
    underlying_index: Optional[str] = None

    @field_validator("creation_timestamp", mode="before")
    @classmethod
    def coerce_ts(cls, v: object) -> int:
        return int(v)

    @field_validator("underlying_price", mode="before")
    @classmethod
    def coerce_price(cls, v: object) -> float:
        if v is None:
            return float("nan")
        return float(v)


class SignalRecord(BaseModel):
    """One trading signal produced by the sandbox calculator."""
    signal_ts: pd.Timestamp
    asset: str
    strategy: str          # straddle | vertical_spread
    direction: str         # long | short
    expiry: pd.Timestamp
    # straddle fields
    strike: Optional[float] = None
    # vertical spread fields
    strike_long: Optional[float] = None
    strike_short: Optional[float] = None
    option_type: Optional[str] = None   # C | P  for spreads

    model_config = {"arbitrary_types_allowed": True}
