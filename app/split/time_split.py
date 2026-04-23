"""Chronological train/validation/test splitting for time-series data.

Design:
- Final test set = last `test_days` calendar days of the dataset.
- Training set = everything before the test set.
- Walk-forward validation inside training:
    Each fold has a train window of `train_window_days`, then a gap of
    `gap_days`, then a validation window of `val_window_days`.
    Folds are generated with `step_days` stride.

All splits are strictly chronological; no shuffling.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Split:
    """One walk-forward fold or the final test split."""
    fold: int                     # 0-based; -1 means final test
    train_start: pd.Timestamp
    train_end: pd.Timestamp       # inclusive
    val_start: pd.Timestamp
    val_end: pd.Timestamp         # inclusive
    is_test: bool = False


def make_splits(
    df: pd.DataFrame,
    test_days: int,
    train_window_days: int,
    val_window_days: int,
    gap_days: int,
    step_days: int,
    timestamp_col: str = "timestamp",
) -> tuple[list[Split], pd.Timestamp]:
    """
    Partition *df* into walk-forward folds + one final test split.

    Returns
    -------
    folds:        List of Split objects (walk-forward folds over training data).
    test_cutoff:  Timestamp where the test set begins (train data ends before this).
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    ts = df[timestamp_col]
    global_start = ts.min()
    global_end = ts.max()

    test_cutoff = global_end - pd.Timedelta(days=test_days)
    train_end_ts = test_cutoff - pd.Timedelta(days=1)

    folds: list[Split] = []
    fold_idx = 0
    cursor = global_start

    while True:
        fold_train_start = cursor
        fold_train_end = cursor + pd.Timedelta(days=train_window_days - 1)
        fold_val_start = fold_train_end + pd.Timedelta(days=gap_days + 1)
        fold_val_end = fold_val_start + pd.Timedelta(days=val_window_days - 1)

        # Stop when the validation window would reach into the test set
        if fold_val_end >= test_cutoff:
            break
        # Stop when training window exceeds available training data
        if fold_train_end > train_end_ts:
            break

        folds.append(Split(
            fold=fold_idx,
            train_start=fold_train_start,
            train_end=fold_train_end,
            val_start=fold_val_start,
            val_end=fold_val_end,
        ))

        cursor += pd.Timedelta(days=step_days)
        fold_idx += 1

    return folds, test_cutoff


def get_fold_data(
    df: pd.DataFrame,
    split: Split,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice *df* into (train_df, val_df) for the given fold."""
    ts = df[timestamp_col]
    train_mask = (ts >= split.train_start) & (ts <= split.train_end)
    val_mask = (ts >= split.val_start) & (ts <= split.val_end)
    return df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def get_test_data(
    df: pd.DataFrame,
    test_cutoff: pd.Timestamp,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) using the test cutoff timestamp."""
    ts = df[timestamp_col]
    return (
        df[ts < test_cutoff].reset_index(drop=True),
        df[ts >= test_cutoff].reset_index(drop=True),
    )


def quick_split(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    days: int = 30,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return a *days*-long window starting at *start_date* for quick runs."""
    start = pd.Timestamp(start_date, tz="UTC") if isinstance(start_date, str) else start_date
    end = start + pd.Timedelta(days=days - 1)
    ts = df[timestamp_col]
    return df[(ts >= start) & (ts <= end)].reset_index(drop=True)
