"""Tests for the core simulator."""
import math

import pandas as pd
import pytest

from app.data.schema import ExecutionModel
from app.sim.pricing import ExecConfig, FeeConfig
from app.sim.simulator import run_simulation


def _base_options_df() -> pd.DataFrame:
    ts0 = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    ts1 = pd.Timestamp("2025-01-02 10:00", tz="UTC")
    ts3 = pd.Timestamp("2025-01-04 10:00", tz="UTC")
    expiry = pd.Timestamp("2025-01-31 08:00", tz="UTC")

    rows = []
    for ts in [ts0, ts1, ts3]:
        for opt_type in ["C", "P"]:
            name = f"BTC-31JAN25-90000-{opt_type}"
            rows.append({
                "timestamp": ts,
                "instrument_name": name,
                "asset": "BTC",
                "expiry_ts": expiry,
                "strike": 90000.0,
                "option_type": opt_type,
                "mark_price": 0.05,
                "mark_iv": 0.75,
                "underlying_price": 90000.0,
                "best_bid_price": 0.048,
                "best_ask_price": 0.052,
                "mid_price": 0.050,
                "open_interest": 100.0,
                "volume": 50.0,
            })
    return pd.DataFrame(rows)


def _base_signal(strategy: str = "straddle") -> pd.DataFrame:
    ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    expiry = pd.Timestamp("2025-01-31 08:00", tz="UTC")
    row = {
        "signal_ts": ts,
        "asset": "BTC",
        "strategy": strategy,
        "direction": "long",
        "expiry": expiry,
        "strike": 90000.0 if strategy == "straddle" else float("nan"),
        "strike_long": float("nan") if strategy == "straddle" else 90000.0,
        "strike_short": float("nan") if strategy == "straddle" else 95000.0,
        "option_type": float("nan") if strategy == "straddle" else "C",
    }
    return pd.DataFrame([row])


class TestStraddle:
    def test_produces_result(self):
        options_df = _base_options_df()
        signals = _base_signal("straddle")
        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1, 3])
        assert len(results) == 2  # one per horizon

    def test_pnl_is_finite(self):
        options_df = _base_options_df()
        signals = _base_signal("straddle")
        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        valid = results[~results["is_nan"]]
        if not valid.empty:
            assert valid["pnl"].apply(lambda x: not math.isnan(x)).all()

    def test_fees_positive(self):
        options_df = _base_options_df()
        signals = _base_signal("straddle")
        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        valid = results[~results["is_nan"]]
        if not valid.empty:
            assert (valid["fees"] >= 0).all()


class TestVerticalSpread:
    def test_missing_second_strike_returns_nan(self):
        """Vertical spread with only one strike in snapshot → NaN."""
        options_df = _base_options_df()  # only strike=90000
        signals = _base_signal("vertical_spread")
        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        # strike_short=95000 doesn't exist → should be NaN
        assert results["is_nan"].all()


class TestEmptySignals:
    def test_empty_signals_returns_empty_df(self):
        options_df = _base_options_df()
        signals = pd.DataFrame(columns=[
            "signal_ts", "asset", "strategy", "direction", "expiry",
            "strike", "strike_long", "strike_short", "option_type",
        ])
        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        assert results.empty


class TestExecutionModels:
    def test_mark_price_used(self):
        options_df = _base_options_df()
        signals = _base_signal("straddle")
        exec_cfg = ExecConfig(model=ExecutionModel.MARK)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        assert len(results) > 0

    def test_crossed_bid_ask(self):
        options_df = _base_options_df()
        signals = _base_signal("straddle")
        exec_cfg = ExecConfig(
            model=ExecutionModel.CROSSED_BID_ASK_PLUS_SLIPPAGE,
            slippage_bps=10.0,
        )
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)
        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        assert len(results) > 0
