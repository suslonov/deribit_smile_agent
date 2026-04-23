"""Tests for chronological splitting logic."""
import pandas as pd
import pytest

from app.split.time_split import (
    Split,
    get_fold_data,
    get_test_data,
    make_splits,
    quick_split,
)


def _make_df(start: str, end: str, freq: str = "1D") -> pd.DataFrame:
    """Create a simple DataFrame with a timestamp column."""
    ts = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "value": range(len(ts))})


class TestMakeSplits:
    def test_returns_folds_and_cutoff(self):
        df = _make_df("2023-01-01", "2024-12-31")
        folds, cutoff = make_splits(
            df=df,
            test_days=90,
            train_window_days=60,
            val_window_days=14,
            gap_days=3,
            step_days=14,
        )
        assert len(folds) > 0
        assert isinstance(cutoff, pd.Timestamp)

    def test_test_cutoff_is_at_end(self):
        df = _make_df("2023-01-01", "2024-12-31")
        _, cutoff = make_splits(df=df, test_days=90, train_window_days=60,
                                val_window_days=14, gap_days=3, step_days=14)
        expected_cutoff = df["timestamp"].max() - pd.Timedelta(days=90)
        assert abs((cutoff - expected_cutoff).days) <= 1

    def test_no_val_in_test_set(self):
        df = _make_df("2023-01-01", "2024-12-31")
        folds, cutoff = make_splits(df=df, test_days=90, train_window_days=60,
                                    val_window_days=14, gap_days=3, step_days=14)
        for fold in folds:
            assert fold.val_end < cutoff, f"Fold {fold.fold} val_end {fold.val_end} >= test_cutoff {cutoff}"

    def test_train_before_val(self):
        df = _make_df("2023-01-01", "2024-12-31")
        folds, _ = make_splits(df=df, test_days=90, train_window_days=60,
                               val_window_days=14, gap_days=3, step_days=14)
        for fold in folds:
            assert fold.train_end < fold.val_start

    def test_gap_respected(self):
        df = _make_df("2023-01-01", "2024-12-31")
        gap = 3
        folds, _ = make_splits(df=df, test_days=90, train_window_days=60,
                               val_window_days=14, gap_days=gap, step_days=14)
        for fold in folds:
            actual_gap = (fold.val_start - fold.train_end).days - 1
            assert actual_gap >= gap

    def test_folds_monotonically_increasing(self):
        df = _make_df("2023-01-01", "2024-12-31")
        folds, _ = make_splits(df=df, test_days=90, train_window_days=60,
                               val_window_days=14, gap_days=3, step_days=14)
        for i in range(1, len(folds)):
            assert folds[i].train_start > folds[i - 1].train_start

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            make_splits(pd.DataFrame(columns=["timestamp"]),
                        test_days=90, train_window_days=60,
                        val_window_days=14, gap_days=3, step_days=14)


class TestGetFoldData:
    def test_slices_correctly(self):
        df = _make_df("2023-01-01", "2024-12-31")
        folds, _ = make_splits(df=df, test_days=90, train_window_days=60,
                               val_window_days=14, gap_days=3, step_days=14)
        fold = folds[0]
        train, val = get_fold_data(df, fold)

        assert (train["timestamp"] >= fold.train_start).all()
        assert (train["timestamp"] <= fold.train_end).all()
        assert (val["timestamp"] >= fold.val_start).all()
        assert (val["timestamp"] <= fold.val_end).all()

    def test_no_overlap_between_train_and_val(self):
        df = _make_df("2023-01-01", "2024-12-31")
        folds, _ = make_splits(df=df, test_days=90, train_window_days=60,
                               val_window_days=14, gap_days=3, step_days=14)
        for fold in folds:
            train, val = get_fold_data(df, fold)
            merged = pd.concat([train["timestamp"], val["timestamp"]])
            assert merged.nunique() == len(train) + len(val)


class TestGetTestData:
    def test_splits_at_cutoff(self):
        df = _make_df("2023-01-01", "2024-12-31")
        _, cutoff = make_splits(df=df, test_days=90, train_window_days=60,
                                val_window_days=14, gap_days=3, step_days=14)
        train, test = get_test_data(df, cutoff)
        assert (train["timestamp"] < cutoff).all()
        assert (test["timestamp"] >= cutoff).all()
        assert len(train) + len(test) == len(df)


class TestQuickSplit:
    def test_returns_correct_window(self):
        df = _make_df("2023-01-01", "2024-12-31")
        result = quick_split(df, "2024-01-01", days=30)
        assert (result["timestamp"] >= pd.Timestamp("2024-01-01", tz="UTC")).all()
        assert (result["timestamp"] <= pd.Timestamp("2024-01-30", tz="UTC")).all()
