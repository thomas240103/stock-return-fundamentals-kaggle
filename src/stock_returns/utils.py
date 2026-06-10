"""General utility helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_file(path: str | Path) -> Path:
    """Return a path if it exists, otherwise raise a clear error."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Required file not found: {resolved}. "
            "Download the Kaggle data and place it in data/raw/."
        )
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return its path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable formatting."""
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def utc_timestamp() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
