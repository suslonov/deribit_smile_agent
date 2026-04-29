"""Load raw Deribit option data from the archive folder.

Archive layout:
    <root>/YYYY-MM-DD/<obfuscated_file>  (many files per day)

Each file is: 1-byte prefix + zlib-compressed Python pickle → list[dict]

Loader flow:
1. Enumerate daily folders in the requested [start_date, end_date] range.
2. For each day, check for a cached parquet file; if absent, read raw files.
3. Combine into one sorted DataFrame.
4. Optionally filter by asset.
"""
from __future__ import annotations

import logging
import os
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data.normalize import normalize_records
from app.data.schema import OPTIONS_COLUMNS

logger = logging.getLogger(__name__)


def load_date_range(
    root: str | Path,
    start: date,
    end: date,
    assets: Optional[list[str]] = None,
    cache_dir: Optional[str | Path] = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Load and return a normalized options DataFrame for [start, end] (inclusive).

    Parameters
    ----------
    root:        Path to the archive root folder (contains YYYY-MM-DD subfolders).
    start/end:   Date range (inclusive).
    assets:      If given, only keep rows for these assets (e.g. ['BTC', 'ETH']).
    cache_dir:   If given, read/write per-day parquet caches here.
    max_workers: Parallel workers for raw file ingestion.
    """
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir else None

    days = _date_range(start, end)
    frames: list[pd.DataFrame] = []

    for day in days:
        df = _load_day(root, day, cache_dir, max_workers)
        if df.empty:
            logger.debug("No data for %s", day)
            continue
        if assets:
            df = df[df["asset"].isin(assets)]
            if df.empty:
                continue
        frames.append(df)

    if not frames:
        logger.warning("No data loaded for range %s – %s", start, end)
        return pd.DataFrame(columns=OPTIONS_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    del frames
    combined.sort_values("timestamp", inplace=True, ignore_index=True)

    logger.info(
        "Loaded %d rows for %s – %s (assets=%s)",
        len(combined), start, end, assets,
    )
    return combined


def _load_day(
    root: Path,
    day: date,
    cache_dir: Optional[Path],
    max_workers: int,
) -> pd.DataFrame:
    """Load one calendar day, using cache if available."""
    day_str = day.strftime("%Y-%m-%d")

    if cache_dir:
        cache_file = cache_dir / f"{day_str}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

    day_path = root / day_str
    if not day_path.exists():
        return pd.DataFrame(columns=OPTIONS_COLUMNS)

    raw_files = [day_path / f for f in os.listdir(day_path) if not f.startswith(".")]
    if not raw_files:
        return pd.DataFrame(columns=OPTIONS_COLUMNS)

    frames = _read_files_parallel(raw_files, max_workers)
    if not frames:
        return pd.DataFrame(columns=OPTIONS_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("timestamp", inplace=True, ignore_index=True)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        logger.debug("Cached %s → %s (%d rows)", day_str, cache_file, len(df))

    return df


def _read_files_parallel(files: list[Path], max_workers: int) -> list[pd.DataFrame]:
    """Read many raw snapshot files using a thread pool (I/O bound work)."""
    frames: list[pd.DataFrame] = []
    if max_workers <= 1:
        for f in files:
            df = _read_single_file(f)
            if df is not None and not df.empty:
                frames.append(df)
        return frames

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = {executor.submit(_read_single_file, f): f for f in files}
        for fut in as_completed(futs):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.debug("Failed to read %s: %s", futs[fut], exc)
    return frames


def _read_single_file(path: Path) -> pd.DataFrame | None:
    """Read one raw snapshot file → normalized DataFrame or None on error."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None

    records = _decompress_pickle(raw)
    if records is None:
        return None
    return normalize_records(records)


def _decompress_pickle(raw: bytes) -> list[dict] | None:
    """Try known decompression strategies: 1-byte skip + zlib, plain zlib."""
    for skip in (1, 0, 2):
        if len(raw) <= skip:
            continue
        try:
            decompressed = zlib.decompress(raw[skip:])
            obj = pickle.loads(decompressed)  # noqa: S301 – trusted internal archive
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


def _date_range(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days
