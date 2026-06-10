"""Ablation study for feature blocks on the temporal validation split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_returns.config import PREFERRED_ENSEMBLE_MODELS, RANDOM_STATE, TARGET_COL
from stock_returns.data import load_train
from stock_returns.ensemble import (
    search_ensemble_weights,
    tune_shrinkage,
    weighted_average,
)
from stock_returns.features import make_feature_frame
from stock_returns.models import build_models, clip_target
from stock_returns.train import fit_named_models, predict_named_models
from stock_returns.utils import ensure_dir
from stock_returns.validation import get_time_split, rmse


ABLATION_STAGES = [
    {
        "stage": "base features only",
        "feature_set": "base",
        "estimator": "primary_model",
        "use_ensemble": False,
    },
    {
        "stage": "base + ranks",
        "feature_set": "ranks",
        "estimator": "primary_model",
        "use_ensemble": False,
    },
    {
        "stage": "base + ranks + composite scores",
        "feature_set": "scores",
        "estimator": "primary_model",
        "use_ensemble": False,
    },
    {
        "stage": "base + ranks + composite scores + ensemble",
        "feature_set": "scores",
        "estimator": "validation_tuned_ensemble",
        "use_ensemble": True,
    },
]


def _split_features(
    train_df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_mask, validation_mask = get_time_split(train_df)
    train_part = train_df.loc[train_mask].copy()
    validation_part = train_df.loc[validation_mask].copy()
    X_train = make_feature_frame(train_part, feature_set=feature_set)
    X_validation = make_feature_frame(validation_part, fit_columns=list(X_train.columns), feature_set=feature_set)
    y_train = pd.to_numeric(train_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_validation = pd.to_numeric(validation_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    return X_train, X_validation, y_train, y_validation


def _evaluate_primary_model(
    train_df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    include_optional: bool,
    random_state: int,
) -> dict[str, Any]:
    X_train, X_validation, y_train, y_validation = _split_features(train_df, feature_set)
    y_train_clipped, _ = clip_target(y_train)
    models = build_models(random_state=random_state, include_optional=include_optional)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {sorted(models)}")

    model = models[model_name]
    model.fit(X_train, y_train_clipped)
    prediction = np.asarray(model.predict(X_validation), dtype=float)
    return {
        "model_name": model_name,
        "n_features": int(X_train.shape[1]),
        "validation_rmse": rmse(y_validation, prediction),
        "raw_ensemble_rmse": np.nan,
        "shrinkage_alpha": np.nan,
        "selected_weights": "",
    }


def _evaluate_ensemble(
    train_df: pd.DataFrame,
    feature_set: str,
    include_optional: bool,
    random_state: int,
) -> dict[str, Any]:
    X_train, X_validation, y_train, y_validation = _split_features(train_df, feature_set)
    y_train_clipped, _ = clip_target(y_train)
    train_mean = float(np.nanmean(y_train))

    models = build_models(random_state=random_state, include_optional=include_optional)
    fitted = fit_named_models(models, X_train, y_train_clipped)
    predictions = predict_named_models(fitted, X_validation)
    ensemble_candidates = {name: predictions[name] for name in PREFERRED_ENSEMBLE_MODELS if name in predictions}
    ensemble_candidates = ensemble_candidates or predictions

    weights, raw_rmse = search_ensemble_weights(ensemble_candidates, y_validation)
    raw_prediction = weighted_average(ensemble_candidates, weights)
    alpha, shrunk_rmse = tune_shrinkage(y_validation, raw_prediction, train_mean=train_mean)

    return {
        "model_name": "validation_tuned_ensemble",
        "n_features": int(X_train.shape[1]),
        "validation_rmse": shrunk_rmse,
        "raw_ensemble_rmse": raw_rmse,
        "shrinkage_alpha": alpha,
        "selected_weights": json.dumps(weights, sort_keys=True),
    }


def run_ablation(
    train_path: str | Path,
    output_path: str | Path = "outputs/ablation_results.csv",
    primary_model: str = "gradient_boosting",
    include_optional: bool = False,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run feature-block ablation and save validation RMSE results."""
    train_df = load_train(train_path)
    rows: list[dict[str, Any]] = []

    for stage in ABLATION_STAGES:
        if stage["use_ensemble"]:
            result = _evaluate_ensemble(
                train_df,
                feature_set=str(stage["feature_set"]),
                include_optional=include_optional,
                random_state=random_state,
            )
        else:
            result = _evaluate_primary_model(
                train_df,
                feature_set=str(stage["feature_set"]),
                model_name=primary_model,
                include_optional=include_optional,
                random_state=random_state,
            )

        rows.append(
            {
                "stage": stage["stage"],
                "feature_set": stage["feature_set"],
                "estimator": result["model_name"],
                "n_features": result["n_features"],
                "validation_rmse": result["validation_rmse"],
                "raw_ensemble_rmse": result["raw_ensemble_rmse"],
                "shrinkage_alpha": result["shrinkage_alpha"],
                "selected_weights": result["selected_weights"],
            }
        )

    results = pd.DataFrame(rows)
    results["improvement_vs_previous"] = results["validation_rmse"].shift(1) - results["validation_rmse"]
    results["improved_vs_previous"] = results["improvement_vs_previous"] > 0

    output = Path(output_path)
    ensure_dir(output.parent)
    results.to_csv(output, index=False)
    return results
