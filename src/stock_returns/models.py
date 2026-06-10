"""Model definitions and target handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_returns.config import RANDOM_STATE, SECTOR_COL


def median_imputer() -> SimpleImputer:
    """Median imputer that keeps all-empty columns stable across temporal splits."""
    return SimpleImputer(strategy="median", keep_empty_features=True)


def clip_target(
    y: pd.Series | np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[np.ndarray, dict[str, float]]:
    """Clip the target by quantiles and return clipped values plus bounds."""
    values = np.asarray(y, dtype=float)
    lower = float(np.nanquantile(values, lower_q))
    upper = float(np.nanquantile(values, upper_q))
    return np.clip(values, lower, upper), {"lower": lower, "upper": upper}


@dataclass
class GlobalMeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline that predicts the training mean."""

    value_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "GlobalMeanRegressor":
        self.value_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("GlobalMeanRegressor is not fitted.")
        return np.full(len(X), self.value_, dtype=float)


@dataclass
class GlobalMedianRegressor(BaseEstimator, RegressorMixin):
    """Baseline that predicts the training median."""

    value_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "GlobalMedianRegressor":
        self.value_ = float(np.nanmedian(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("GlobalMedianRegressor is not fitted.")
        return np.full(len(X), self.value_, dtype=float)


class SectorMedianRegressor(BaseEstimator, RegressorMixin):
    """Predict each sector's historical median target with global fallback."""

    def __init__(self, sector_col: str = SECTOR_COL) -> None:
        self.sector_col = sector_col
        self.global_median_: float | None = None
        self.sector_medians_: dict[float, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "SectorMedianRegressor":
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_median_ = float(target.median())
        if self.sector_col in X.columns:
            medians = target.groupby(X[self.sector_col]).median()
            self.sector_medians_ = {float(k): float(v) for k, v in medians.items()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_median_ is None:
            raise RuntimeError("SectorMedianRegressor is not fitted.")
        if self.sector_col not in X.columns:
            return np.full(len(X), self.global_median_, dtype=float)
        sectors = pd.to_numeric(X[self.sector_col], errors="coerce")
        return sectors.map(self.sector_medians_).fillna(self.global_median_).to_numpy(dtype=float)


class RankRidgeRegressor(BaseEstimator, RegressorMixin):
    """Ridge regression using rank, z-score, score, and sector columns."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.columns_: list[str] = []
        self.pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "RankRidgeRegressor":
        rank_like = [
            col
            for col in X.columns
            if "rank" in col or col.endswith("_sector_z") or col.endswith("_score") or col == SECTOR_COL
        ]
        self.columns_ = rank_like or list(X.columns)
        self.pipeline_ = Pipeline(
            steps=[
                ("imputer", median_imputer()),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline_.fit(X[self.columns_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("RankRidgeRegressor is not fitted.")
        return self.pipeline_.predict(X[self.columns_])


def _hist_gradient_boosting(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", median_imputer()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.03,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _extra_trees(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", median_imputer()),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _optional_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {}

    try:
        from lightgbm import LGBMRegressor

        models["lightgbm"] = Pipeline(
            steps=[
                ("imputer", median_imputer()),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", median_imputer()),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        models["catboost"] = Pipeline(
            steps=[
                ("imputer", median_imputer()),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.03,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=random_state,
                        verbose=False,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    return models


def build_models(random_state: int = RANDOM_STATE, include_optional: bool = True) -> dict[str, object]:
    """Build all available models."""
    models: dict[str, object] = {
        "global_mean": GlobalMeanRegressor(),
        "global_median": GlobalMedianRegressor(),
        "sector_median": SectorMedianRegressor(),
        "gradient_boosting": _hist_gradient_boosting(random_state),
        "extra_trees": _extra_trees(random_state),
        "ridge_rank": RankRidgeRegressor(alpha=10.0),
    }
    if include_optional:
        models.update(_optional_models(random_state))
    return models
