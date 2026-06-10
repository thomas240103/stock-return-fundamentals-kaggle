"""Prediction and submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from stock_returns.config import FINAL_PREDICTION_CLIP, ID_COL, PREFERRED_ENSEMBLE_MODELS, RANDOM_STATE, TARGET_COL
from stock_returns.data import load_test, load_train
from stock_returns.ensemble import (
    apply_shrinkage,
    default_weights_for,
    search_ensemble_weights,
    tune_shrinkage,
    weighted_average,
)
from stock_returns.features import make_feature_frame
from stock_returns.models import build_models, clip_target
from stock_returns.train import fit_named_models, predict_named_models
from stock_returns.utils import ensure_dir
from stock_returns.validation import get_time_split


def validate_submission_frame(submission: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate Kaggle submission shape and columns."""
    expected_columns = [ID_COL, TARGET_COL]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns}.")
    if len(submission) != len(test_df):
        raise ValueError(f"Submission row count {len(submission)} does not match test row count {len(test_df)}.")
    if submission[TARGET_COL].isna().any():
        raise ValueError("Submission contains NaN predictions.")
    if not submission[ID_COL].reset_index(drop=True).equals(test_df[ID_COL].reset_index(drop=True)):
        raise ValueError("Submission ids do not match test ids in order.")


def _tune_on_temporal_split(
    train_df: pd.DataFrame,
    include_optional: bool,
    random_state: int,
    feature_set: str,
) -> dict[str, Any]:
    train_mask, validation_mask = get_time_split(train_df)
    train_part = train_df.loc[train_mask].copy()
    validation_part = train_df.loc[validation_mask].copy()

    X_train = make_feature_frame(train_part, feature_set=feature_set)
    X_validation = make_feature_frame(validation_part, fit_columns=list(X_train.columns), feature_set=feature_set)
    y_train_real = pd.to_numeric(train_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_validation_real = pd.to_numeric(validation_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_train_clipped, _ = clip_target(y_train_real)

    models = build_models(random_state=random_state, include_optional=include_optional)
    fitted = fit_named_models(models, X_train, y_train_clipped)
    validation_predictions = predict_named_models(fitted, X_validation)
    ensemble_candidates = {name: validation_predictions[name] for name in PREFERRED_ENSEMBLE_MODELS if name in validation_predictions}
    ensemble_candidates = ensemble_candidates or validation_predictions

    weights, _ = search_ensemble_weights(ensemble_candidates, y_validation_real)
    raw_prediction = weighted_average(ensemble_candidates, weights)
    train_mean = float(np.nanmean(y_train_real))
    alpha, _ = tune_shrinkage(y_validation_real, raw_prediction, train_mean=train_mean)
    return {"weights": weights, "alpha": alpha}


def fit_full_train_bundle(
    train_df: pd.DataFrame,
    include_optional: bool = True,
    random_state: int = RANDOM_STATE,
    feature_set: str = "scores",
) -> dict[str, Any]:
    """Tune on the temporal split, then refit models on all train rows."""
    try:
        tuning = _tune_on_temporal_split(
            train_df,
            include_optional=include_optional,
            random_state=random_state,
            feature_set=feature_set,
        )
    except ValueError:
        tuning = {"weights": None, "alpha": 1.0}

    X_full = make_feature_frame(train_df, feature_set=feature_set)
    y_full_real = pd.to_numeric(train_df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_full_clipped, target_clip = clip_target(y_full_real)
    train_mean = float(np.nanmean(y_full_real))

    models = build_models(random_state=random_state, include_optional=include_optional)
    fitted = fit_named_models(models, X_full, y_full_clipped)

    return {
        "models": fitted,
        "feature_columns": list(X_full.columns),
        "target_clip": target_clip,
        "train_mean": train_mean,
        "ensemble_weights": tuning["weights"],
        "shrinkage_alpha": tuning["alpha"],
        "random_state": random_state,
        "feature_set": feature_set,
    }


def predict_with_bundle(bundle: dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    """Predict test rows with a fitted model bundle."""
    feature_set = bundle.get("feature_set", "scores")
    X_test = make_feature_frame(test_df, fit_columns=bundle["feature_columns"], feature_set=feature_set)
    predictions = predict_named_models(bundle["models"], X_test)
    ensemble_candidates = {name: predictions[name] for name in PREFERRED_ENSEMBLE_MODELS if name in predictions}
    ensemble_candidates = ensemble_candidates or predictions
    weights = bundle.get("ensemble_weights") or default_weights_for(ensemble_candidates)
    raw_prediction = weighted_average(ensemble_candidates, weights)
    shrunk = apply_shrinkage(
        raw_prediction,
        train_mean=float(bundle["train_mean"]),
        alpha=float(bundle.get("shrinkage_alpha", 1.0)),
    )
    low, high = FINAL_PREDICTION_CLIP
    return np.clip(shrunk, low, high)


def make_submission(
    train_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    include_optional: bool = True,
    random_state: int = RANDOM_STATE,
    model_output_path: str | Path | None = None,
    feature_set: str = "scores",
) -> pd.DataFrame:
    """Fit on train data and write a Kaggle submission CSV."""
    train_df = load_train(train_path)
    test_df = load_test(test_path)
    bundle = fit_full_train_bundle(
        train_df,
        include_optional=include_optional,
        random_state=random_state,
        feature_set=feature_set,
    )
    prediction = predict_with_bundle(bundle, test_df)

    submission = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: prediction})
    validate_submission_frame(submission, test_df)

    target = Path(output_path)
    ensure_dir(target.parent)
    submission.to_csv(target, index=False)

    if model_output_path is not None:
        model_target = Path(model_output_path)
        ensure_dir(model_target.parent)
        joblib.dump(bundle, model_target)

    return submission


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a joblib model bundle."""
    return joblib.load(path)
