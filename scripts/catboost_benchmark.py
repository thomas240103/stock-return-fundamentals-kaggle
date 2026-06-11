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
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.config import ID_COL, SECTOR_COL, TARGET_COL
from stock_returns.models import clip_target
from stock_returns.utils import ensure_dir
from stock_returns.validation import get_time_split, percent_improvement


EXCLUDE_COLUMNS = {ID_COL, "ticker", TARGET_COL, "period_start", "period_end", "start_year", SECTOR_COL}
SUBMISSION_TRANSFORMS = [
    "raw",
    "shrink_0p8",
    "clip_100",
    "clip_200",
    "clip_300",
    "clip_500",
    "shrink_0p8_clip_100",
    "shrink_0p8_clip_200",
    "shrink_0p8_clip_300",
    "shrink_0p8_clip_500",
    "mean_shrink_0p8_clip_100",
    "mean_shrink_0p8_clip_200",
    "mean_shrink_0p8_clip_300",
    "mean_shrink_0p8_clip_500",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-safe CatBoost benchmark inspired by public leaderboard notebooks.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--test", default=None, help="Optional path to Kaggle test.csv.")
    parser.add_argument("--output-dir", default="outputs/catboost_benchmark", help="Directory for benchmark outputs.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--use-year-offset",
        action="store_true",
        help="Add relative year features where the earliest available start_year is encoded as 0.",
    )
    parser.add_argument("--make-submission", action="store_true", help="Refit on all train rows and write submission.csv.")
    parser.add_argument(
        "--submission-target",
        choices=["raw", "clip_1_99", "clip_5_95"],
        default="clip_1_99",
        help="Target preprocessing used when refitting on all train rows for submission.",
    )
    parser.add_argument(
        "--submission-transform",
        choices=SUBMISSION_TRANSFORMS,
        default="clip_100",
        help="Final prediction transform for submission. Defaults to the best 2022 validation transform.",
    )
    parser.add_argument(
        "--write-variant-submissions",
        action="store_true",
        help="Also write multiple transformed submission files from the same fitted test predictions.",
    )
    parser.add_argument(
        "--variant-transforms",
        default="clip_100,clip_200,clip_300,clip_500,raw,shrink_0p8_clip_100,shrink_0p8_clip_200,shrink_0p8_clip_300",
        help="Comma-separated transforms to write when --write-variant-submissions is used.",
    )
    return parser.parse_args()


def require_catboost():
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor
    except ImportError as exc:
        raise ImportError("Install CatBoost first: python -m pip install catboost") from exc


def load_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "start_year" not in df.columns and "period_start" in df.columns:
        df["start_year"] = pd.to_datetime(df["period_start"], errors="coerce").dt.year
    return df


def infer_year_zero(*frames: pd.DataFrame) -> float:
    """Use the earliest available year as time zero across train/valid/test frames."""
    years = []
    for frame in frames:
        if "start_year" in frame.columns:
            values = pd.to_numeric(frame["start_year"], errors="coerce").dropna()
            if not values.empty:
                years.append(float(values.min()))
    if not years:
        return 0.0
    return float(min(years))


def add_year_offset_features(df: pd.DataFrame, year_zero: float, last_train_year: float | None = None) -> pd.DataFrame:
    """Encode calendar time as offsets from the earliest year instead of raw years."""
    out = df.copy()
    if "start_year" not in out.columns:
        return out
    years = pd.to_numeric(out["start_year"], errors="coerce")
    offset = years - float(year_zero)
    out["year_offset"] = offset
    out["year_offset_squared"] = offset**2
    if last_train_year is not None:
        out["year_offset_from_last_train_year"] = years - float(last_train_year)
    return out


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(a, errors="coerce")
    denominator = pd.to_numeric(b, errors="coerce")
    return numerator.where(np.isfinite(numerator)) / denominator.where(np.isfinite(denominator) & (denominator != 0))


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in {ID_COL, "ticker", "period_start", "period_end"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ey_inverse_pe"] = safe_divide(pd.Series(1.0, index=out.index), out["pe_ttm"])
    out["ni_rev"] = safe_divide(out["net_income_ttm"], out["revenue_ttm"])
    out["ni_ast"] = safe_divide(out["net_income_ttm"], out["total_assets"])
    out["roe_to_inverse_pe"] = safe_divide(out["roe"], out["ey_inverse_pe"])
    out["operating_x_net_margin"] = out["operating_margin"] * out["net_margin"]
    out["pe_to_price_book"] = safe_divide(out["pe_ttm"], out["price_to_book"])
    out["asset_turnover"] = safe_divide(out["revenue_ttm"], out["total_assets"])
    out["working_capital"] = out["current_assets"] - out["current_liabilities"]
    out["working_capital_to_assets"] = safe_divide(out["working_capital"], out["total_assets"])
    out["long_term_debt_to_assets"] = safe_divide(out["long_term_debt"], out["total_assets"])
    out["goodwill_to_assets"] = safe_divide(out["goodwill"], out["total_assets"])
    out["peg_proxy"] = safe_divide(out["pe_ttm"], out["revenue_growth_yoy"])
    out["quality_score_raw"] = out["roe"] * out["net_margin"]
    out["value_profit_raw"] = out["ey_inverse_pe"] * out["roe"]
    return out


def sector_reference(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    refs: dict[str, pd.DataFrame] = {}
    for col in columns:
        if col not in df.columns:
            continue
        grouped = df.groupby(SECTOR_COL)[col]
        refs[col] = pd.DataFrame({"median": grouped.median(), "std": grouped.std().replace(0, np.nan)})
    return refs


def add_sector_relative_features(df: pd.DataFrame, refs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    for col, ref in refs.items():
        sec_median = out[SECTOR_COL].map(ref["median"])
        sec_std = out[SECTOR_COL].map(ref["std"])
        out[f"{col}_vs_sec_train"] = safe_divide(out[col] - sec_median, sec_std + 1e-8)
        out[f"{col}_to_sec_median_train"] = safe_divide(out[col], sec_median)
    return out


def target_variant(y: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, float | None]]:
    if name == "raw":
        return y.copy(), {"lower": None, "upper": None}
    if name == "clip_1_99":
        clipped, bounds = clip_target(y, 0.01, 0.99)
        return clipped, bounds
    if name == "clip_5_95":
        clipped, bounds = clip_target(y, 0.05, 0.95)
        return clipped, bounds
    raise ValueError(f"Unknown target variant: {name}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_prediction(y_true: np.ndarray, pred: np.ndarray, train_mean: float) -> dict[str, float]:
    raw_rmse = rmse(y_true, pred)
    shrunk = pred * 0.8
    clipped = np.clip(pred, -100.0, 100.0)
    shrunk_clipped = np.clip(shrunk, -100.0, 100.0)
    mean_shrunk = train_mean + 0.8 * (pred - train_mean)
    mean_shrunk_clipped = np.clip(mean_shrunk, -100.0, 100.0)
    return {
        "rmse_raw": raw_rmse,
        "rmse_shrink_0p8": rmse(y_true, shrunk),
        "rmse_clip_100": rmse(y_true, clipped),
        "rmse_shrink_0p8_clip_100": rmse(y_true, shrunk_clipped),
        "rmse_mean_shrink_0p8_clip_100": rmse(y_true, mean_shrunk_clipped),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


def transform_submission_prediction(pred: np.ndarray, train_mean: float, transform: str) -> np.ndarray:
    values = np.asarray(pred, dtype=float)
    if transform == "raw":
        return values
    if transform == "shrink_0p8":
        return values * 0.8
    if transform.startswith("clip_"):
        upper = float(transform.removeprefix("clip_"))
        return np.clip(values, -100.0, upper)
    if transform.startswith("shrink_0p8_clip_"):
        upper = float(transform.removeprefix("shrink_0p8_clip_"))
        return np.clip(values * 0.8, -100.0, upper)
    if transform.startswith("mean_shrink_0p8_clip_"):
        upper = float(transform.removeprefix("mean_shrink_0p8_clip_"))
        return np.clip(float(train_mean) + 0.8 * (values - float(train_mean)), -100.0, upper)
    raise ValueError(f"Unknown submission transform: {transform}")


def summarize_prediction(prediction: np.ndarray) -> dict[str, float]:
    values = np.asarray(prediction, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])]


def fit_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    args: argparse.Namespace,
) -> Any:
    CatBoostRegressor = require_catboost()
    model = CatBoostRegressor(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
        verbose=0,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )
    eval_set = (X_valid, y_valid) if X_valid is not None and y_valid is not None else None
    model.fit(X_train, y_train, eval_set=eval_set)
    return model


def run_validation(train_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    train_mask, valid_mask = get_time_split(train_df)
    year_zero = infer_year_zero(train_df)
    last_train_year = float(pd.to_numeric(train_df.loc[train_mask, "start_year"], errors="coerce").max())
    base_train = add_basic_features(train_df)
    if args.use_year_offset:
        base_train = add_year_offset_features(base_train, year_zero=year_zero, last_train_year=last_train_year)
    ref_cols = [
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
    refs = sector_reference(base_train.loc[train_mask], ref_cols)
    engineered = add_sector_relative_features(base_train, refs)
    features = feature_columns(engineered)

    X_train_raw = engineered.loc[train_mask, features].astype("float32").to_numpy()
    X_valid_raw = engineered.loc[valid_mask, features].astype("float32").to_numpy()
    y_train_real = pd.to_numeric(engineered.loc[train_mask, TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_valid = pd.to_numeric(engineered.loc[valid_mask, TARGET_COL], errors="coerce").to_numpy(dtype=float)
    reference_rmse = rmse(y_valid, np.full(len(y_valid), float(np.mean(y_train_real))))

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_valid = imputer.transform(X_valid_raw)

    rows = []
    for target_name in ["raw", "clip_1_99", "clip_5_95"]:
        y_train, bounds = target_variant(y_train_real, target_name)
        model = fit_catboost(X_train, y_train, X_valid, y_valid, args)
        pred_valid = np.asarray(model.predict(X_valid), dtype=float)
        row = {
            "target_preprocessing": target_name,
            "n_features": len(features),
            "use_year_offset": args.use_year_offset,
            "year_zero": year_zero,
            "last_train_year": last_train_year,
            "target_lower": bounds["lower"],
            "target_upper": bounds["upper"],
        }
        row.update(evaluate_prediction(y_valid, pred_valid, train_mean=float(np.mean(y_train_real))))
        best_rmse = min(value for key, value in row.items() if key.startswith("rmse_"))
        row["best_rmse"] = best_rmse
        row["best_improvement_vs_train_mean_pct"] = percent_improvement(reference_rmse, best_rmse)
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("best_rmse")
    results.to_csv(output_dir / "catboost_benchmark_results.csv", index=False)
    return results


def make_submission(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> Path:
    year_zero = infer_year_zero(train_df, test_df)
    last_train_year = float(pd.to_numeric(train_df["start_year"], errors="coerce").max())
    train_features = add_basic_features(train_df)
    test_features = add_basic_features(test_df)
    if args.use_year_offset:
        train_features = add_year_offset_features(train_features, year_zero=year_zero, last_train_year=last_train_year)
        test_features = add_year_offset_features(test_features, year_zero=year_zero, last_train_year=last_train_year)
    ref_cols = [
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
    refs = sector_reference(train_features, ref_cols)
    train_features = add_sector_relative_features(train_features, refs)
    test_features = add_sector_relative_features(test_features, refs)
    features = feature_columns(train_features)
    for col in features:
        if col not in test_features.columns:
            test_features[col] = np.nan

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_features[features].astype("float32").to_numpy())
    X_test = imputer.transform(test_features[features].astype("float32").to_numpy())
    y_train = pd.to_numeric(train_features[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    y_train_model, _ = target_variant(y_train, args.submission_target)

    model = fit_catboost(X_train, y_train_model, None, None, args)
    raw_pred = np.asarray(model.predict(X_test), dtype=float)
    train_mean = float(np.mean(y_train))
    pred = transform_submission_prediction(raw_pred, train_mean=train_mean, transform=args.submission_transform)
    submission = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: pred})
    output_path = output_dir / "submission_catboost_benchmark.csv"
    submission.to_csv(output_path, index=False)

    summaries = [
        {
            "submission_file": output_path.name,
            "target_preprocessing": args.submission_target,
            "transform": args.submission_transform,
            "use_year_offset": args.use_year_offset,
            "year_zero": year_zero,
            "last_train_year": last_train_year,
            **summarize_prediction(pred),
        }
    ]
    if args.write_variant_submissions:
        requested = [item.strip() for item in args.variant_transforms.split(",") if item.strip()]
        invalid = sorted(set(requested) - set(SUBMISSION_TRANSFORMS))
        if invalid:
            raise ValueError(f"Unknown variant transforms: {invalid}")
        for transform in requested:
            variant_pred = transform_submission_prediction(raw_pred, train_mean=train_mean, transform=transform)
            variant_name = f"submission_{args.submission_target}_{transform}.csv"
            variant_path = output_dir / variant_name
            pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: variant_pred}).to_csv(variant_path, index=False)
            summaries.append(
                {
                    "submission_file": variant_name,
                    "target_preprocessing": args.submission_target,
                    "transform": transform,
                    "use_year_offset": args.use_year_offset,
                    "year_zero": year_zero,
                    "last_train_year": last_train_year,
                    **summarize_prediction(variant_pred),
                }
            )
    pd.DataFrame(summaries).to_csv(output_dir / "submission_variant_summaries.csv", index=False)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    train_df = load_frame(args.train)
    results = run_validation(train_df, output_dir, args)
    print(results.to_string(index=False))

    if args.make_submission:
        test_path = Path(args.test) if args.test else Path(args.train).with_name("test.csv")
        test_df = load_frame(test_path)
        output_path = make_submission(train_df, test_df, output_dir, args)
        print(f"Saved submission to {output_path}")

    print(f"Saved benchmark results to {output_dir / 'catboost_benchmark_results.csv'}")


if __name__ == "__main__":
    main()
