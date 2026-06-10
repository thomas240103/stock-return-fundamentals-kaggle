"""Training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from stock_returns.config import DEFAULT_MAX_MISSING_FRACTION, ID_COL, PREFERRED_ENSEMBLE_MODELS, RANDOM_STATE, TARGET_COL
from stock_returns.data import load_train
from stock_returns.ensemble import (
    apply_shrinkage,
    default_weights_for,
    search_ensemble_weights,
    tune_shrinkage,
    weighted_average,
)
from stock_returns.features import make_feature_frame, select_columns_by_missingness
from stock_returns.models import build_models, clip_target
from stock_returns.utils import ensure_dir, save_json, utc_timestamp
from stock_returns.validation import get_time_split, rmse


def fit_named_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, object]:
    """Fit a dictionary of sklearn-compatible models."""
    fitted: dict[str, object] = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        fitted[name] = model.fit(X, y)
    return fitted


def predict_named_models(models: dict[str, object], X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predict with each fitted model."""
    return {name: np.asarray(model.predict(X), dtype=float) for name, model in models.items()}


def _core_ensemble_predictions(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    selected = {name: predictions[name] for name in PREFERRED_ENSEMBLE_MODELS if name in predictions}
    return selected or predictions


def train_validation_pipeline(
    train_path: str | Path,
    output_dir: str | Path = "outputs",
    include_optional: bool = True,
    random_state: int = RANDOM_STATE,
    feature_set: str = "scores",
    max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
) -> dict[str, Any]:
    """Train models on 2019-2021 and validate on 2022."""
    output_path = ensure_dir(output_dir)
    models_path = ensure_dir(output_path / "models")

    train_df = load_train(train_path)
    train_mask, validation_mask = get_time_split(train_df)

    train_part = train_df.loc[train_mask].copy()
    validation_part = train_df.loc[validation_mask].copy()

    X_train_full = make_feature_frame(train_part, feature_set=feature_set)
    feature_columns, dropped_missing_columns = select_columns_by_missingness(
        X_train_full,
        max_missing_fraction=max_missing_fraction,
    )
    X_train = X_train_full[feature_columns]
    X_validation = make_feature_frame(validation_part, fit_columns=feature_columns, feature_set=feature_set)
    y_train_real = pd.to_numeric(train_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_validation_real = pd.to_numeric(validation_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_train_clipped, target_clip = clip_target(y_train_real)
    train_mean = float(np.nanmean(y_train_real))

    models = build_models(random_state=random_state, include_optional=include_optional)
    fitted = fit_named_models(models, X_train, y_train_clipped)
    validation_predictions = predict_named_models(fitted, X_validation)

    metrics: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_validation)),
        "n_features": int(len(feature_columns)),
        "feature_set": feature_set,
        "max_missing_fraction": max_missing_fraction,
        "dropped_missing_columns": dropped_missing_columns,
        "target_clip": target_clip,
        "models": {},
    }
    for name, pred in validation_predictions.items():
        score = rmse(y_validation_real, pred)
        metrics["models"][name] = {"validation_rmse": score}
        print(f"{name}: validation RMSE = {score:.6f}")

    ensemble_candidates = _core_ensemble_predictions(validation_predictions)
    tuned_weights, tuned_rmse = search_ensemble_weights(ensemble_candidates, y_validation_real)
    weights = tuned_weights or default_weights_for(ensemble_candidates)
    ensemble_raw = weighted_average(ensemble_candidates, weights)
    ensemble_raw_rmse = rmse(y_validation_real, ensemble_raw)
    alpha, shrunk_rmse = tune_shrinkage(y_validation_real, ensemble_raw, train_mean=train_mean)
    ensemble_shrunk = apply_shrinkage(ensemble_raw, train_mean=train_mean, alpha=alpha)

    metrics["ensemble"] = {
        "tuned_weights": tuned_weights,
        "tuned_rmse": tuned_rmse,
        "selected_weights": weights,
        "selected_weights_source": "validation_search",
        "raw_rmse": ensemble_raw_rmse,
        "shrinkage_alpha": alpha,
        "shrunk_rmse": shrunk_rmse,
        "train_mean": train_mean,
    }
    print(f"ensemble raw: validation RMSE = {ensemble_raw_rmse:.6f}")
    print(f"ensemble shrunk: validation RMSE = {shrunk_rmse:.6f} (alpha={alpha:.2f})")

    validation_output = pd.DataFrame(
        {
            ID_COL: validation_part[ID_COL].to_numpy() if ID_COL in validation_part.columns else np.arange(len(validation_part)),
            "y_true": y_validation_real,
            "ensemble_raw": ensemble_raw,
            "ensemble_shrunk": ensemble_shrunk,
        }
    )
    for name, pred in validation_predictions.items():
        validation_output[f"pred_{name}"] = pred
    validation_output.to_csv(output_path / "validation_predictions.csv", index=False)

    bundle = {
        "models": fitted,
        "feature_columns": list(X_train.columns),
        "dropped_missing_columns": dropped_missing_columns,
        "max_missing_fraction": max_missing_fraction,
        "target_clip": target_clip,
        "train_mean": train_mean,
        "ensemble_weights": weights,
        "shrinkage_alpha": alpha,
        "random_state": random_state,
        "feature_set": feature_set,
        "metrics": metrics,
    }
    joblib.dump(bundle, models_path / "model_bundle.joblib")
    save_json(metrics, output_path / "metrics.json")
    save_json(
        {
            "feature_columns": list(X_train.columns),
            "feature_set": feature_set,
            "dropped_missing_columns": dropped_missing_columns,
            "max_missing_fraction": max_missing_fraction,
            "ensemble_weights": weights,
            "shrinkage_alpha": alpha,
            "target_clip": target_clip,
        },
        models_path / "metadata.json",
    )

    return bundle
