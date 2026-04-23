"""Time utilities."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_now_str() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")
