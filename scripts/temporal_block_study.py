from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

from catboost_benchmark import (  # noqa: E402
    add_basic_features,
    add_sector_relative_features,
    add_year_offset_features,
    feature_columns,
    fit_catboost,
    infer_year_zero,
    sector_reference,
    target_variant,
    transform_submission_prediction,
)
from stock_returns.config import TARGET_COL  # noqa: E402
from stock_returns.utils import ensure_dir  # noqa: E402
from stock_returns.validation import percent_improvement, rmse_as_target_std_pct  # noqa: E402


REF_COLS = [
    "pe_ttm",
    "price_to_book",
    "price_to_sales",
    "roe",
    "roa",
    "net_margin",
    "gross_margin",
    "operating_margin",
    "revenue_growth_yoy",
    "revenue_growth_3y",
    "ey_inverse_pe",
    "asset_turnover",
    "debt_to_equity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study temporal block stability for the CatBoost benchmark.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--output-dir", default="outputs/temporal_block_study", help="Directory for study outputs.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--target-variants",
        default="raw,clip_1_99,clip_5_95",
        help="Comma-separated target preprocessing variants to test.",
    )
    parser.add_argument(
        "--prediction-transforms",
        default="raw,clip_100,clip_200,clip_300,shrink_0p8_clip_100,shrink_0p8_clip_200",
        help="Comma-separated validation prediction transforms to test.",
    )
    parser.add_argument(
        "--no-relative-year-features",
        action="store_true",
        help="Disable relative time features such as year offset from the first train year.",
    )
    return parser.parse_args()


def load_train(path: str | Path) -> pd.DataFrame:
    train = pd.read_csv(path)
    if "start_year" not in train.columns and "period_start" in train.columns:
        train["start_year"] = pd.to_datetime(train["period_start"], errors="coerce").dt.year
    train["start_year"] = pd.to_numeric(train["start_year"], errors="coerce")
    return train


def temporal_splits(years: list[int]) -> list[dict[str, Any]]:
    available = sorted(years)
    splits: list[dict[str, Any]] = []

    def add_split(split_type: str, train_years: list[int], valid_year: int) -> None:
        if not train_years or valid_year not in available:
            return
        train_years = sorted(train_years)
        if any(year not in available for year in train_years):
            return
        horizon = valid_year - max(train_years)
        first_gap = valid_year - min(train_years)
        splits.append(
            {
                "split_name": f"{split_type}_{min(train_years)}_{max(train_years)}_to_{valid_year}",
                "train_years": train_years,
                "valid_year": valid_year,
                "split_type": split_type,
                "horizon_years": horizon,
                "gap_from_first_train_year": first_gap,
                "train_span_years": max(train_years) - min(train_years) + 1,
            }
        )

    for train_year in available:
        add_split("one_year_single", [train_year], train_year + 1)
        add_split("two_year_single", [train_year], train_year + 2)

    for valid_year in available[1:]:
        add_split("expanding", [year for year in available if year < valid_year], valid_year)
        add_split("rolling1", [valid_year - 1], valid_year)
        add_split("rolling2", [valid_year - 2, valid_year - 1], valid_year)

    return splits


def rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, prediction)))


def prediction_stats(prediction: np.ndarray) -> dict[str, float]:
    values = np.asarray(prediction, dtype=float)
    return {
        "pred_mean": float(np.mean(values)),
        "pred_std": float(np.std(values)),
        "pred_min": float(np.min(values)),
        "pred_p95": float(np.quantile(values, 0.95)),
        "pred_p99": float(np.quantile(values, 0.99)),
        "pred_max": float(np.max(values)),
    }


def evaluate_split(
    train_df: pd.DataFrame,
    split: dict[str, Any],
    target_variants: list[str],
    prediction_transforms: list[str],
    year_zero: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    train_mask = train_df["start_year"].isin(split["train_years"])
    valid_mask = train_df["start_year"].eq(split["valid_year"])
    train_part = train_df.loc[train_mask].copy()
    valid_part = train_df.loc[valid_mask].copy()

    base_train = add_basic_features(train_part)
    base_valid = add_basic_features(valid_part)
    refs = sector_reference(base_train, REF_COLS)
    train_features = add_sector_relative_features(base_train, refs)
    valid_features = add_sector_relative_features(base_valid, refs)
    use_relative_year = not args.no_relative_year_features
    last_train_year = max(split["train_years"])
    train_features = add_year_offset_features(
        train_features,
        year_zero=year_zero,
        last_train_year=last_train_year,
    )
    valid_features = add_year_offset_features(
        valid_features,
        year_zero=year_zero,
        last_train_year=last_train_year,
    )
    if not use_relative_year:
        relative_cols = ["year_offset", "year_offset_squared", "year_offset_from_last_train_year"]
        train_features = train_features.drop(columns=relative_cols, errors="ignore")
        valid_features = valid_features.drop(columns=relative_cols, errors="ignore")
    features = feature_columns(train_features)
    for col in features:
        if col not in valid_features.columns:
            valid_features[col] = np.nan

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_features[features].astype("float32").to_numpy())
    X_valid = imputer.transform(valid_features[features].astype("float32").to_numpy())
    y_train_real = pd.to_numeric(train_features[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_valid = pd.to_numeric(valid_features[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    train_mean = float(np.mean(y_train_real))
    reference_rmse = rmse(y_valid, np.full(len(y_valid), train_mean))

    rows: list[dict[str, Any]] = []
    for target_name in target_variants:
        y_train_model, bounds = target_variant(y_train_real, target_name)
        model = fit_catboost(X_train, y_train_model, X_valid, y_valid, args)
        raw_pred = np.asarray(model.predict(X_valid), dtype=float)
        for transform in prediction_transforms:
            pred = transform_submission_prediction(raw_pred, train_mean=train_mean, transform=transform)
            score = rmse(y_valid, pred)
            row = {
                "split_name": split["split_name"],
                "split_type": split["split_type"],
                "train_years": ",".join(str(year) for year in split["train_years"]),
                "valid_year": split["valid_year"],
                "horizon_years": split["horizon_years"],
                "gap_from_first_train_year": split["gap_from_first_train_year"],
                "train_span_years": split["train_span_years"],
                "relative_year_features": use_relative_year,
                "year_zero": year_zero,
                "last_train_year": last_train_year,
                "target_preprocessing": target_name,
                "prediction_transform": transform,
                "n_train": int(len(train_features)),
                "n_valid": int(len(valid_features)),
                "n_features": int(len(features)),
                "train_target_mean": train_mean,
                "valid_target_mean": float(np.mean(y_valid)),
                "valid_target_std": float(np.std(y_valid)),
                "target_lower": bounds["lower"],
                "target_upper": bounds["upper"],
                "reference_train_mean_rmse": reference_rmse,
                "validation_rmse": score,
                "improvement_vs_train_mean_pct": percent_improvement(reference_rmse, score),
                "rmse_as_validation_target_std_pct": rmse_as_target_std_pct(y_valid, score),
            }
            row.update(prediction_stats(pred))
            rows.append(row)
    return rows


def write_markdown(output_dir: Path, results: pd.DataFrame) -> Path:
    report_path = output_dir / "temporal_block_study.md"
    best_by_split = (
        results.sort_values("validation_rmse")
        .groupby("split_name", as_index=False)
        .head(1)
        .sort_values(["valid_year", "split_type"])
    )
    transform_summary = (
        results.groupby(["target_preprocessing", "prediction_transform"], as_index=False)
        .agg(
            mean_rmse=("validation_rmse", "mean"),
            median_rmse=("validation_rmse", "median"),
            mean_improvement_pct=("improvement_vs_train_mean_pct", "mean"),
            wins=("validation_rmse", lambda x: int((x == x.min()).sum())),
        )
        .sort_values("mean_rmse")
    )

    def frame_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows available._"
        lines = [
            "| " + " | ".join(str(col) for col in df.columns) + " |",
            "| " + " | ".join(["---"] * len(df.columns)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
        return "\n".join(lines)

    lines = [
        "# Temporal Block Study",
        "",
        "This study checks whether the CatBoost benchmark is stable across time blocks, not just on the 2022 validation split.",
        "",
        "## Best Variant By Split",
        "",
        frame_to_markdown(
            best_by_split[
                [
                    "split_name",
                    "train_years",
                    "valid_year",
                    "horizon_years",
                    "gap_from_first_train_year",
                    "train_span_years",
                    "year_zero",
                    "last_train_year",
                    "target_preprocessing",
                    "prediction_transform",
                    "validation_rmse",
                    "improvement_vs_train_mean_pct",
                    "valid_target_mean",
                    "valid_target_std",
                    "pred_mean",
                    "pred_max",
                ]
            ].round(4)
        ),
        "",
        "## Average Variant Stability",
        "",
        frame_to_markdown(transform_summary.head(20).round(4)),
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    train_df = load_train(args.train)
    year_zero = infer_year_zero(train_df)
    years = sorted(int(year) for year in train_df["start_year"].dropna().unique())
    target_variants = [item.strip() for item in args.target_variants.split(",") if item.strip()]
    prediction_transforms = [item.strip() for item in args.prediction_transforms.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for split in temporal_splits(years):
        rows.extend(evaluate_split(train_df, split, target_variants, prediction_transforms, year_zero, args))

    results = pd.DataFrame(rows).sort_values(["valid_year", "validation_rmse"])
    results.to_csv(output_dir / "temporal_block_results.csv", index=False)
    report_path = write_markdown(output_dir, results)

    best = results.groupby("split_name", as_index=False).head(1)
    print(best.to_string(index=False))
    print(f"Saved temporal block study to {output_dir}")
    print(f"Open {report_path}")


if __name__ == "__main__":
    main()
