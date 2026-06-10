"""Temporal validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_time_split(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return train and validation masks for the official temporal split."""
    if "start_year" not in train_df.columns:
        raise ValueError("Expected column 'start_year' for temporal validation.")

    years = pd.to_numeric(train_df["start_year"], errors="coerce")
    train_mask = years < 2022
    validation_mask = years == 2022

    if not train_mask.any():
        raise ValueError("Temporal split produced no training rows with start_year < 2022.")
    if not validation_mask.any():
        raise ValueError("Temporal split produced no validation rows with start_year == 2022.")

    return train_mask, validation_mask


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Compute root mean squared error."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape[0] != pred.shape[0]:
        raise ValueError(f"Shape mismatch: y_true has {true.shape[0]} rows, y_pred has {pred.shape[0]} rows.")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def percent_improvement(reference_score: float, candidate_score: float) -> float:
    """Return percentage improvement for a lower-is-better score."""
    if not np.isfinite(reference_score) or reference_score <= 0:
        return float("nan")
    return float((reference_score - candidate_score) / reference_score * 100.0)


def rmse_as_target_std_pct(y_true: np.ndarray | pd.Series, score: float) -> float:
    """Return RMSE as a percentage of the validation target standard deviation."""
    target = np.asarray(y_true, dtype=float)
    target_std = float(np.nanstd(target))
    if not np.isfinite(target_std) or target_std <= 0:
        return float("nan")
    return float(score / target_std * 100.0)
