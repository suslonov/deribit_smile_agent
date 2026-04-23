"""Tests for the sandbox calculator API contract."""
import pandas as pd
import pytest

from sandbox.runner import SandboxRunner, SandboxError


def _minimal_options_df() -> pd.DataFrame:
    ts = pd.Timestamp("2025-01-01 10:00", tz="UTC")
    expiry = pd.Timestamp("2025-01-31 08:00", tz="UTC")
    rows = []
    for opt_type in ["C", "P"]:
        rows.append({
            "timestamp": ts,
            "instrument_name": f"BTC-31JAN25-90000-{opt_type}",
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


def _minimal_config() -> dict:
    return {
        "calculator": {
            "assets": ["BTC"],
            "rv_window": 20,
            "iv_rv_threshold": 0.05,
            "min_oi": 1.0,
            "min_strikes": 1,
            "target_dte_min": 1,
            "target_dte_max": 90,
            "straddle_moneyness_tol": 0.05,
            "smile_slope_threshold": 0.0,
        }
    }


class TestSandboxAPIContract:
    def setup_method(self):
        self.runner = SandboxRunner(
            calculator_path="sandbox/calculator.py",
            timeout_seconds=30,
            allowed_imports=frozenset({
                "__future__",
                "numpy", "pandas", "scipy", "math", "statistics",
                "itertools", "functools", "collections", "typing",
                "datetime", "re", "copy", "warnings",
            }),
            verify_determinism=True,
        )
        self.runner.reload()

    def test_returns_dataframe(self):
        df = _minimal_options_df()
        config = _minimal_config()
        result = self.runner.run(df, config["calculator"])
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self):
        required = {
            "signal_ts", "asset", "strategy", "direction", "expiry",
            "strike", "strike_long", "strike_short", "option_type",
        }
        df = _minimal_options_df()
        config = _minimal_config()
        result = self.runner.run(df, config["calculator"])
        assert required.issubset(set(result.columns))

    def test_deterministic_output(self):
        df = _minimal_options_df()
        config = _minimal_config()
        r1 = self.runner.run(df, config["calculator"], seed=42)
        r2 = self.runner.run(df, config["calculator"], seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_does_not_mutate_input(self):
        df = _minimal_options_df()
        original_hash = pd.util.hash_pandas_object(df).sum()
        config = _minimal_config()
        self.runner.run(df, config["calculator"])
        assert pd.util.hash_pandas_object(df).sum() == original_hash

    def test_validates_source_passes(self):
        violations = self.runner.validate_source()
        assert violations == [], f"Unexpected violations: {violations}"


class TestSandboxValidation:
    def test_rejects_os_import(self, tmp_path):
        bad_source = '''\
import os
import pandas as pd

def compute(options_df, config, seed=42):
    os.listdir("/")
    return pd.DataFrame()
'''
        calc_path = tmp_path / "bad_calculator.py"
        calc_path.write_text(bad_source)
        runner = SandboxRunner(
            calculator_path=calc_path,
            allowed_imports=frozenset({"pandas", "numpy"}),
        )
        runner.reload()
        violations = runner.validate_source()
        assert any("os" in v for v in violations)

    def test_rejects_open_call(self, tmp_path):
        bad_source = '''\
import pandas as pd

def compute(options_df, config, seed=42):
    with open("/etc/passwd") as f:
        pass
    return pd.DataFrame()
'''
        calc_path = tmp_path / "bad_calc2.py"
        calc_path.write_text(bad_source)
        runner = SandboxRunner(
            calculator_path=calc_path,
            allowed_imports=frozenset({"pandas", "numpy"}),
        )
        runner.reload()
        violations = runner.validate_source()
        assert any("open" in v for v in violations)
