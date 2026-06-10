"""Prediction ensembling and shrinkage."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from stock_returns.config import DEFAULT_ENSEMBLE_WEIGHTS, PREFERRED_ENSEMBLE_MODELS
from stock_returns.validation import rmse


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative weights to sum to one."""
    cleaned = {name: max(0.0, float(weight)) for name, weight in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        equal = 1.0 / max(len(cleaned), 1)
        return {name: equal for name in cleaned}
    return {name: weight / total for name, weight in cleaned.items()}


def default_weights_for(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    """Return neutral fallback weights restricted to available predictions."""
    available = {name: weight for name, weight in DEFAULT_ENSEMBLE_WEIGHTS.items() if name in predictions}
    if available:
        return normalize_weights(available)
    return normalize_weights({name: 1.0 for name in predictions})


def weighted_average(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Compute a weighted average of named predictions."""
    if not predictions:
        raise ValueError("No predictions were provided for ensembling.")
    normalized = normalize_weights({name: weights.get(name, 0.0) for name in predictions})
    result = None
    for name, pred in predictions.items():
        weighted = normalized[name] * np.asarray(pred, dtype=float)
        result = weighted if result is None else result + weighted
    return np.asarray(result, dtype=float)


def search_ensemble_weights(
    predictions: dict[str, np.ndarray],
    y_true: pd.Series | np.ndarray,
    step: float = 0.1,
    max_models: int = 4,
) -> tuple[dict[str, float], float]:
    """Search a small non-negative simplex for validation weights."""
    if not predictions:
        raise ValueError("No predictions were provided for weight search.")

    names = [name for name in PREFERRED_ENSEMBLE_MODELS if name in predictions]
    if len(names) < 2:
        names = list(predictions)
    names = names[:max_models]

    grid_values = np.round(np.arange(0.0, 1.0 + step, step), 10)
    best_weights = default_weights_for({name: predictions[name] for name in names})
    best_score = rmse(y_true, weighted_average(predictions, best_weights))

    for combo in product(grid_values, repeat=len(names)):
        total = float(np.sum(combo))
        if not np.isclose(total, 1.0):
            continue
        weights = {name: float(weight) for name, weight in zip(names, combo)}
        candidate = weighted_average(predictions, weights)
        score = rmse(y_true, candidate)
        if score < best_score:
            best_score = score
            best_weights = weights

    return normalize_weights(best_weights), float(best_score)


def apply_shrinkage(prediction: np.ndarray, train_mean: float, alpha: float) -> np.ndarray:
    """Shrink predictions toward the training mean."""
    pred = np.asarray(prediction, dtype=float)
    return alpha * pred + (1.0 - alpha) * float(train_mean)


def tune_shrinkage(
    y_true: pd.Series | np.ndarray,
    prediction: np.ndarray,
    train_mean: float,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Tune shrinkage alpha on validation data."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)

    best_alpha = 1.0
    best_score = rmse(y_true, prediction)
    for alpha in grid:
        candidate = apply_shrinkage(prediction, train_mean=train_mean, alpha=float(alpha))
        score = rmse(y_true, candidate)
        if score < best_score:
            best_alpha = float(alpha)
            best_score = float(score)
    return best_alpha, best_score
