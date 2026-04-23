"""Deterministic hashing utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(path: str | Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dict(obj: dict[str, Any]) -> str:
    """Return SHA-256 hex digest of a JSON-serialised dict (sorted keys)."""
    serialised = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


def short_hash(full_hash: str, length: int = 8) -> str:
    return full_hash[:length]
