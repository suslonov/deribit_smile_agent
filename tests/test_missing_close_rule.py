"""Tests for the missing-close rule in the simulator."""
import pandas as pd
import pytest

from app.sim.simulator import MAX_GAP_DAYS, _get_snapshot_at_or_after


def _make_options_df(timestamps: list[str]) -> pd.DataFrame:
    rows = []
    for ts_str in timestamps:
        ts = pd.Timestamp(ts_str, tz="UTC")
        rows.append({
            "timestamp": ts,
            "instrument_name": "BTC-31JAN25-90000-C",
            "asset": "BTC",
            "expiry_ts": pd.Timestamp("2025-01-31 08:00", tz="UTC"),
            "strike": 90000.0,
            "option_type": "C",
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


class TestGetSnapshotAtOrAfter:
    def test_returns_exact_match(self):
        df = _make_options_df(["2025-01-01 10:00", "2025-01-01 10:01"])
        result = _get_snapshot_at_or_after(
            df.sort_values("timestamp"),
            pd.Timestamp("2025-01-01 10:00", tz="UTC"),
        )
        assert not result.empty
        assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 10:00", tz="UTC")

    def test_returns_next_available(self):
        df = _make_options_df(["2025-01-01 10:05", "2025-01-01 10:06"])
        result = _get_snapshot_at_or_after(
            df.sort_values("timestamp"),
            pd.Timestamp("2025-01-01 10:00", tz="UTC"),
        )
        assert not result.empty
        assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 10:05", tz="UTC")

    def test_returns_empty_when_no_data_after(self):
        df = _make_options_df(["2025-01-01 09:00"])
        result = _get_snapshot_at_or_after(
            df.sort_values("timestamp"),
            pd.Timestamp("2025-01-01 10:00", tz="UTC"),
        )
        assert result.empty


class TestMissingCloseRule:
    """Verify that the simulator applies the 3-day gap rule correctly."""

    def test_nan_when_close_gap_exceeded(self):
        from app.sim.simulator import run_simulation, MAX_GAP_DAYS
        from app.sim.pricing import ExecConfig, FeeConfig
        from app.data.schema import ExecutionModel

        # Create a scenario where the close snapshot is > MAX_GAP_DAYS after target
        signal_ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        # Only one snapshot at open time, and next one is 5 days later
        timestamps = [
            "2025-01-01 10:00",   # open
            "2025-01-06 10:00",   # close candidate (5 days > MAX_GAP_DAYS=3)
        ]
        options_df = _make_options_df(timestamps)
        # Add second instrument for straddle
        put_rows = options_df.copy()
        put_rows["instrument_name"] = "BTC-31JAN25-90000-P"
        put_rows["option_type"] = "P"
        options_df = pd.concat([options_df, put_rows], ignore_index=True)

        signals = pd.DataFrame([{
            "signal_ts": signal_ts,
            "asset": "BTC",
            "strategy": "straddle",
            "direction": "long",
            "expiry": pd.Timestamp("2025-01-31 08:00", tz="UTC"),
            "strike": 90000.0,
            "strike_long": float("nan"),
            "strike_short": float("nan"),
            "option_type": float("nan"),
        }])

        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)

        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        # Horizon 1: target = Jan 2, first available = Jan 6 → gap=4 days > MAX_GAP_DAYS
        horizon_1 = results[results["horizon_days"] == 1]
        assert not horizon_1.empty
        assert horizon_1["is_nan"].all()
        assert (horizon_1["nan_reason"] == "close_gap_exceeded").all()

    def test_nan_when_expiry_before_close(self):
        from app.sim.simulator import run_simulation
        from app.sim.pricing import ExecConfig, FeeConfig
        from app.data.schema import ExecutionModel

        signal_ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        # Expiry is in 2 days, but horizon is 3 days
        expiry = pd.Timestamp("2025-01-03 08:00", tz="UTC")
        timestamps = ["2025-01-01 10:00", "2025-01-04 10:00"]
        options_df = _make_options_df(timestamps)
        options_df["expiry_ts"] = expiry

        put_rows = options_df.copy()
        put_rows["instrument_name"] = "BTC-31JAN25-90000-P"
        put_rows["option_type"] = "P"
        options_df = pd.concat([options_df, put_rows], ignore_index=True)
        options_df["expiry_ts"] = expiry

        signals = pd.DataFrame([{
            "signal_ts": signal_ts,
            "asset": "BTC",
            "strategy": "straddle",
            "direction": "long",
            "expiry": expiry,
            "strike": 90000.0,
            "strike_long": float("nan"),
            "strike_short": float("nan"),
            "option_type": float("nan"),
        }])

        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)

        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[3])
        horizon_3 = results[results["horizon_days"] == 3]
        assert not horizon_3.empty
        assert horizon_3["is_nan"].all()
        assert (horizon_3["nan_reason"] == "expiry_before_close").all()

    def test_valid_close_when_gap_within_limit(self):
        from app.sim.simulator import run_simulation
        from app.sim.pricing import ExecConfig, FeeConfig
        from app.data.schema import ExecutionModel

        signal_ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
        # Target = Jan 2, close at Jan 3 (gap=1 day < MAX_GAP_DAYS)
        timestamps = ["2025-01-01 10:00", "2025-01-03 10:00"]
        options_df = _make_options_df(timestamps)

        put_rows = options_df.copy()
        put_rows["instrument_name"] = "BTC-31JAN25-90000-P"
        put_rows["option_type"] = "P"
        options_df = pd.concat([options_df, put_rows], ignore_index=True)

        signals = pd.DataFrame([{
            "signal_ts": signal_ts,
            "asset": "BTC",
            "strategy": "straddle",
            "direction": "long",
            "expiry": pd.Timestamp("2025-01-31 08:00", tz="UTC"),
            "strike": 90000.0,
            "strike_long": float("nan"),
            "strike_short": float("nan"),
            "option_type": float("nan"),
        }])

        exec_cfg = ExecConfig(model=ExecutionModel.MID)
        fee_cfg = FeeConfig(pct_of_underlying=0.0003, max_pct_of_option=0.125)

        results = run_simulation(signals, options_df, exec_cfg, fee_cfg, horizons=[1])
        horizon_1 = results[results["horizon_days"] == 1]
        assert not horizon_1.empty
        assert not horizon_1["is_nan"].all()
