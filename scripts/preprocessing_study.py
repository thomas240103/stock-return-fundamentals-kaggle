from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.config import ID_COL, TARGET_COL
from stock_returns.data import load_train
from stock_returns.features import make_feature_frame, select_columns_by_missingness
from stock_returns.models import RankRidgeRegressor
from stock_returns.utils import ensure_dir
from stock_returns.validation import get_time_split, percent_improvement, rmse, rmse_as_target_std_pct


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to train-set quantile bounds before imputation/scaling."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "QuantileClipper":
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(values, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(values, self.upper_q, axis=0)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("QuantileClipper is not fitted.")
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study preprocessing choices on the 2022 temporal validation split.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--output-dir", default="outputs/preprocessing_study", help="Directory for study outputs.")
    parser.add_argument("--max-missing-fraction", type=float, default=0.98, help="Drop features above this train missing fraction.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--feature-sets", default="base,ranks,scores", help="Comma-separated feature sets to test.")
    parser.add_argument(
        "--target-variants",
        default="raw,clip_1_99,clip_2_98,clip_5_95",
        help="Comma-separated target preprocessing variants to test.",
    )
    parser.add_argument("--include-histgb", action="store_true", help="Also test a conservative histogram gradient boosting model.")
    return parser.parse_args()


def target_variant(y: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, float | None]]:
    if name == "raw":
        return y.copy(), {"lower_q": None, "upper_q": None, "lower": None, "upper": None}
    lower_q, upper_q = {
        "clip_0p5_99p5": (0.005, 0.995),
        "clip_1_99": (0.01, 0.99),
        "clip_2_98": (0.02, 0.98),
        "clip_5_95": (0.05, 0.95),
    }[name]
    lower = float(np.nanquantile(y, lower_q))
    upper = float(np.nanquantile(y, upper_q))
    return np.clip(y, lower, upper), {"lower_q": lower_q, "upper_q": upper_q, "lower": lower, "upper": upper}


def build_preprocess_models(random_state: int, include_histgb: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {
        "ridge_standard": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=25.0)),
            ]
        ),
        "ridge_robust": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
                ("model", Ridge(alpha=25.0)),
            ]
        ),
        "ridge_winsor_standard": Pipeline(
            steps=[
                ("feature_clip", QuantileClipper(0.01, 0.99)),
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=25.0)),
            ]
        ),
        "rank_ridge_existing": RankRidgeRegressor(alpha=10.0),
    }
    if include_histgb:
        models["histgb_winsor"] = Pipeline(
            steps=[
                ("feature_clip", QuantileClipper(0.01, 0.99)),
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.025,
                        max_leaf_nodes=15,
                        min_samples_leaf=50,
                        l2_regularization=5.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def prediction_summary(prediction: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=float)
    return {
        "pred_mean": float(np.nanmean(pred)),
        "pred_std": float(np.nanstd(pred)),
        "pred_min": float(np.nanmin(pred)),
        "pred_p01": float(np.nanquantile(pred, 0.01)),
        "pred_p99": float(np.nanquantile(pred, 0.99)),
        "pred_max": float(np.nanmax(pred)),
    }


def feature_diagnostics(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in train_df.select_dtypes(include=[np.number]).columns
        if col not in {ID_COL, TARGET_COL}
    ]
    rows: list[dict[str, float | str]] = []
    for col in numeric_cols:
        train_values = pd.to_numeric(train_df[col], errors="coerce")
        q01 = float(train_values.quantile(0.01))
        q99 = float(train_values.quantile(0.99))
        train_std = float(train_values.std())
        row: dict[str, float | str] = {
            "column": col,
            "train_missing_fraction": float(train_values.isna().mean()),
            "train_mean": float(train_values.mean()),
            "train_std": train_std,
            "train_min": float(train_values.min()),
            "train_q01": q01,
            "train_median": float(train_values.median()),
            "train_q99": q99,
            "train_max": float(train_values.max()),
            "upper_tail_ratio": float(abs(train_values.max()) / (abs(q99) + 1e-9)),
            "lower_tail_ratio": float(abs(train_values.min()) / (abs(q01) + 1e-9)),
        }
        if test_df is not None and col in test_df.columns:
            test_values = pd.to_numeric(test_df[col], errors="coerce")
            pooled_std = np.nanmean([train_values.std(), test_values.std()])
            row.update(
                {
                    "test_missing_fraction": float(test_values.isna().mean()),
                    "missing_fraction_diff": float(test_values.isna().mean() - train_values.isna().mean()),
                    "test_mean": float(test_values.mean()),
                    "standardized_mean_diff": float(
                        (test_values.mean() - train_values.mean()) / pooled_std
                        if pooled_std and np.isfinite(pooled_std)
                        else np.nan
                    ),
                }
            )
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    sort_cols = [col for col in ["standardized_mean_diff", "upper_tail_ratio", "train_missing_fraction"] if col in diagnostics.columns]
    if sort_cols:
        diagnostics["_sort_key"] = diagnostics[sort_cols].abs().max(axis=1)
        diagnostics = diagnostics.sort_values("_sort_key", ascending=False).drop(columns="_sort_key")
    return diagnostics.reset_index(drop=True)


def study_preprocessing(
    train_df: pd.DataFrame,
    output_dir: Path,
    max_missing_fraction: float,
    random_state: int,
    feature_sets: list[str],
    target_variants: list[str],
    include_histgb: bool,
) -> pd.DataFrame:
    train_mask, validation_mask = get_time_split(train_df)
    train_part = train_df.loc[train_mask].copy()
    validation_part = train_df.loc[validation_mask].copy()
    y_train_real = pd.to_numeric(train_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_validation_real = pd.to_numeric(validation_part[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    reference_prediction = np.full(len(y_validation_real), float(np.nanmean(y_train_real)))
    reference_rmse = rmse(y_validation_real, reference_prediction)

    rows: list[dict[str, Any]] = []

    for feature_set in feature_sets:
        X_train_full = make_feature_frame(train_part, feature_set=feature_set)
        feature_columns, dropped_missing_columns = select_columns_by_missingness(
            X_train_full,
            max_missing_fraction=max_missing_fraction,
        )
        X_train = X_train_full[feature_columns]
        X_validation = make_feature_frame(validation_part, fit_columns=feature_columns, feature_set=feature_set)
        for target_name in target_variants:
            y_train, target_bounds = target_variant(y_train_real, target_name)
            for model_name, model in build_preprocess_models(random_state, include_histgb=include_histgb).items():
                try:
                    model.fit(X_train, y_train)
                    prediction = np.asarray(model.predict(X_validation), dtype=float)
                    score = rmse(y_validation_real, prediction)
                    row = {
                        "feature_set": feature_set,
                        "target_preprocessing": target_name,
                        "model": model_name,
                        "validation_rmse": score,
                        "improvement_vs_train_mean_pct": percent_improvement(reference_rmse, score),
                        "rmse_as_validation_target_std_pct": rmse_as_target_std_pct(y_validation_real, score),
                        "n_features": len(feature_columns),
                        "n_dropped_missing_features": len(dropped_missing_columns),
                    }
                    row.update({f"target_{key}": value for key, value in target_bounds.items()})
                    row.update(prediction_summary(prediction))
                    rows.append(row)
                except Exception as exc:  # pragma: no cover - diagnostic script should keep going.
                    rows.append(
                        {
                            "feature_set": feature_set,
                            "target_preprocessing": target_name,
                            "model": model_name,
                            "validation_rmse": np.nan,
                            "error": repr(exc),
                            "n_features": len(feature_columns),
                            "n_dropped_missing_features": len(dropped_missing_columns),
                        }
                    )

    results = pd.DataFrame(rows).sort_values("validation_rmse", ascending=True, na_position="last")
    results.to_csv(output_dir / "preprocessing_results.csv", index=False)
    return results


def write_markdown_summary(output_dir: Path, results: pd.DataFrame, diagnostics: pd.DataFrame) -> Path:
    report_path = output_dir / "preprocessing_study.md"
    top = results.head(15).copy()
    top_cols = [
        "feature_set",
        "target_preprocessing",
        "model",
        "validation_rmse",
        "improvement_vs_train_mean_pct",
        "rmse_as_validation_target_std_pct",
        "pred_std",
        "pred_min",
        "pred_max",
        "n_features",
    ]
    diagnostic_cols = [
        "column",
        "train_missing_fraction",
        "missing_fraction_diff",
        "standardized_mean_diff",
        "upper_tail_ratio",
        "lower_tail_ratio",
    ]

    def frame_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows available._"
        markdown = [
            "| " + " | ".join(str(col) for col in df.columns) + " |",
            "| " + " | ".join(["---"] * len(df.columns)) + " |",
        ]
        for _, row in df.iterrows():
            markdown.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
        return "\n".join(markdown)

    lines = [
        "# Preprocessing Study",
        "",
        "Temporal split: train years `< 2022`, validation year `2022`.",
        "",
        "## Best Validation Results",
        "",
        frame_to_markdown(top[top_cols].round(6)),
        "",
        "## Highest-Risk Feature Diagnostics",
        "",
        frame_to_markdown(
            diagnostics[[col for col in diagnostic_cols if col in diagnostics.columns]]
            .head(20)
            .round(6)
        ),
        "",
        "## Interpretation",
        "",
        "- Large target outliers make RMSE extremely sensitive to a few observations.",
        "- Missingness and train-test drift should be handled with train-fitted preprocessing, missing flags, and conservative shrinkage.",
        "- Prefer validation-tested preprocessing over manually weighted final formulas.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    train_df = load_train(args.train)

    test_path = Path(args.train).with_name("test.csv")
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    diagnostics = feature_diagnostics(train_df, test_df)
    diagnostics.to_csv(output_dir / "feature_diagnostics.csv", index=False)
    results = study_preprocessing(
        train_df=train_df,
        output_dir=output_dir,
        max_missing_fraction=args.max_missing_fraction,
        random_state=args.random_state,
        feature_sets=[item.strip() for item in args.feature_sets.split(",") if item.strip()],
        target_variants=[item.strip() for item in args.target_variants.split(",") if item.strip()],
        include_histgb=args.include_histgb,
    )
    report_path = write_markdown_summary(output_dir, results, diagnostics)

    print(results.head(20).to_string(index=False))
    print(f"Saved preprocessing study to {output_dir}")
    print(f"Open {report_path}")


if __name__ == "__main__":
    main()
