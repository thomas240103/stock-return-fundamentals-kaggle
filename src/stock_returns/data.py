"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_returns.config import ID_COL, TARGET_COL
from stock_returns.utils import ensure_file


def read_csv_checked(path: str | Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV file and verify required columns when provided."""
    csv_path = ensure_file(path)
    df = pd.read_csv(csv_path)
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return df


def load_train(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle train file."""
    return read_csv_checked(path, required_columns=[ID_COL, TARGET_COL, "start_year"])


def load_test(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle test file."""
    return read_csv_checked(path, required_columns=[ID_COL])
