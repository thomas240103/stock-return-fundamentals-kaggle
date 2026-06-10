"""Feature engineering for stock fundamental data.

The main public entry point is `make_feature_frame`. It deliberately removes
the target and identifier columns, handles missing values defensively, and
returns only numeric model-ready columns.

Feature sets:

- `base`: dates, missing flags, signed logs, raw numeric fields, and derived ratios.
- `ranks`: base features plus global and sector-relative rank/z-score features.
- `scores` / `all`: ranks plus literature-informed composite scores.

Composite scores are inputs for ML models, not hand-weighted final predictions:
`value_score`, `quality_score`, `growth_score`, `balance_sheet_score`,
`liquidity_score`, `piotroski_style_score`, and `quality_value_score`.
They are based primarily on sector-relative percentile ranks so firms are
compared against closer economic peers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from stock_returns.config import (
    DATE_COLUMNS,
    DEFAULT_MAX_MISSING_FRACTION,
    EPSILON,
    FORBIDDEN_FEATURE_COLUMNS,
    LOWER_IS_BETTER_COLUMNS,
    RANK_COLUMNS,
    SECTOR_COL,
    SIGNED_LOG_COLUMNS,
    VALUATION_COLUMNS,
)

FeatureSet = Literal["base", "ranks", "scores", "all"]
COMPOSITE_SCORE_COLUMNS = [
    "value_score",
    "quality_score",
    "growth_score",
    "balance_sheet_score",
    "liquidity_score",
    "piotroski_style_score",
    "quality_value_score",
]


def select_columns_by_missingness(
    X: pd.DataFrame,
    max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
) -> tuple[list[str], dict[str, float]]:
    """Select columns with acceptable missingness using training features only.

    Columns above `max_missing_fraction` are excluded. This removes all-empty or
    nearly empty columns that add warning noise and unstable imputations, while
    keeping moderately sparse financial fields available to the model.
    """
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError("max_missing_fraction must be between 0.0 and 1.0.")

    missing_fraction = X.isna().mean()
    keep_columns = missing_fraction[missing_fraction <= max_missing_fraction].index.tolist()
    dropped = missing_fraction[missing_fraction > max_missing_fraction].sort_values(ascending=False)
    if not keep_columns:
        raise ValueError("Missingness filter removed every feature column.")
    return keep_columns, {col: float(value) for col, value in dropped.items()}


def signed_log1p(values: pd.Series) -> pd.Series:
    """Compute sign(x) * log1p(abs(x)) for skewed signed quantities."""
    numeric = pd.to_numeric(values, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").abs()
    return num / (den + EPSILON)


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DATE_COLUMNS:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_month"] = parsed.dt.month
        out[f"{col}_quarter"] = parsed.dt.quarter
        out[f"{col}_year"] = parsed.dt.year
    return out


def _numeric_source_columns(df: pd.DataFrame) -> list[str]:
    ignored = FORBIDDEN_FEATURE_COLUMNS | set(DATE_COLUMNS)
    return [
        col
        for col in df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(df[col])
    ]


def _add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _numeric_source_columns(out):
        out[f"missing_{col}"] = out[col].isna().astype(np.int8)
    return out


def _add_signed_log_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SIGNED_LOG_COLUMNS:
        if col in out.columns:
            out[f"{col}_signed_log1p"] = signed_log1p(out[col])
    return out


def _col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _add_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    revenue = _col_or_nan(out, "revenue_ttm")
    net_income = _col_or_nan(out, "net_income_ttm")
    debt = _col_or_nan(out, "long_term_debt")
    assets = _col_or_nan(out, "total_assets")
    equity = _col_or_nan(out, "stockholders_equity")
    current_assets = _col_or_nan(out, "current_assets")
    current_liabilities = _col_or_nan(out, "current_liabilities")
    market_cap = _col_or_nan(out, "market_cap")
    goodwill = _col_or_nan(out, "goodwill")
    inventory = _col_or_nan(out, "inventory")

    working_capital = current_assets - current_liabilities
    out["net_income_to_revenue"] = _safe_divide(net_income, revenue)
    out["debt_to_assets"] = _safe_divide(debt, assets)
    out["equity_to_assets"] = _safe_divide(equity, assets)
    out["working_capital"] = working_capital
    out["working_capital_to_assets"] = _safe_divide(working_capital, assets)
    out["sales_yield"] = _safe_divide(revenue, market_cap)
    out["earnings_yield"] = _safe_divide(net_income, market_cap)
    out["goodwill_to_assets"] = _safe_divide(goodwill, assets)
    out["inventory_to_assets"] = _safe_divide(inventory, assets)
    return out


def _sector_median_signal(df: pd.DataFrame, col: str, higher_is_better: bool) -> pd.Series:
    values = _col_or_nan(df, col)
    if SECTOR_COL not in df.columns:
        threshold = values.median()
        result = values >= threshold if higher_is_better else values <= threshold
        return result.fillna(False).astype(np.int8)

    sectors = df[SECTOR_COL]
    thresholds = values.groupby(sectors).transform("median")
    fallback = values.median()
    thresholds = thresholds.fillna(fallback)
    result = values >= thresholds if higher_is_better else values <= thresholds
    return result.fillna(False).astype(np.int8)


def _add_piotroski_style_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["positive_roa"] = (_col_or_nan(out, "roa") > 0).fillna(False).astype(np.int8)
    out["positive_roe"] = (_col_or_nan(out, "roe") > 0).fillna(False).astype(np.int8)
    out["positive_net_margin"] = (_col_or_nan(out, "net_margin") > 0).fillna(False).astype(np.int8)
    out["positive_revenue_growth_yoy"] = (_col_or_nan(out, "revenue_growth_yoy") > 0).fillna(False).astype(np.int8)
    out["positive_revenue_growth_3y"] = (_col_or_nan(out, "revenue_growth_3y") > 0).fillna(False).astype(np.int8)
    out["positive_earnings_yield"] = (_col_or_nan(out, "earnings_yield") > 0).fillna(False).astype(np.int8)
    out["low_debt_sector"] = _sector_median_signal(out, "debt_to_equity", higher_is_better=False)
    out["high_current_ratio_sector"] = _sector_median_signal(out, "current_ratio", higher_is_better=True)
    out["high_quick_ratio_sector"] = _sector_median_signal(out, "quick_ratio", higher_is_better=True)

    signal_cols = [
        "positive_roa",
        "positive_roe",
        "positive_net_margin",
        "positive_revenue_growth_yoy",
        "positive_revenue_growth_3y",
        "positive_earnings_yield",
        "low_debt_sector",
        "high_current_ratio_sector",
        "high_quick_ratio_sector",
    ]
    out["fundamental_score"] = out[signal_cols].sum(axis=1)
    return out


def _rank_percentile(values: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", pct=True, ascending=ascending)


def _sector_rank_percentile(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    values = _col_or_nan(df, col)
    if SECTOR_COL not in df.columns:
        return _rank_percentile(values, ascending=ascending)
    return values.groupby(df[SECTOR_COL]).rank(method="average", pct=True, ascending=ascending)


def _sector_z_score(df: pd.DataFrame, col: str) -> pd.Series:
    values = _col_or_nan(df, col)
    if SECTOR_COL not in df.columns:
        std = values.std(ddof=0)
        return (values - values.mean()) / (std + EPSILON)
    grouped = values.groupby(df[SECTOR_COL])
    mean = grouped.transform("mean")
    std = grouped.transform("std").fillna(0.0)
    return (values - mean) / (std + EPSILON)


def _add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RANK_COLUMNS:
        if col not in out.columns:
            continue
        out[f"{col}_rank_all"] = _rank_percentile(out[col], ascending=True)
        out[f"{col}_rank_sector"] = _sector_rank_percentile(out, col, ascending=True)
        out[f"{col}_sector_z"] = _sector_z_score(out, col)

    for col in LOWER_IS_BETTER_COLUMNS:
        if col in out.columns:
            out[f"{col}_low_rank_all"] = _rank_percentile(out[col], ascending=False)
            out[f"{col}_low_rank_sector"] = _sector_rank_percentile(out, col, ascending=False)

    for col in VALUATION_COLUMNS:
        if col in out.columns:
            out[f"{col}_cheap_rank_all"] = out[f"{col}_low_rank_all"]
            out[f"{col}_cheap_rank_sector"] = out[f"{col}_low_rank_sector"]
            out[f"{col}_cheap_rank"] = out[f"{col}_cheap_rank_sector"]
    return out


def _mean_existing_columns(df: pd.DataFrame, columns: list[str], missing_value: float = 0.5) -> pd.Series:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return pd.Series(missing_value, index=df.index, dtype="float64")
    return df[existing].astype("float64").fillna(missing_value).mean(axis=1)


def _add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add literature-informed score blocks from sector-relative percentile ranks."""
    out = df.copy()

    value_components = [
        "pe_ttm_cheap_rank_sector",
        "price_to_book_cheap_rank_sector",
        "price_to_sales_cheap_rank_sector",
        "earnings_yield_rank_sector",
        "sales_yield_rank_sector",
        "dividend_yield_rank_sector",
    ]
    quality_components = [
        "roe_rank_sector",
        "roa_rank_sector",
        "net_margin_rank_sector",
        "gross_margin_rank_sector",
        "operating_margin_rank_sector",
        "net_income_to_revenue_rank_sector",
    ]
    growth_components = [
        "revenue_growth_yoy_rank_sector",
        "revenue_growth_3y_rank_sector",
    ]
    balance_sheet_components = [
        "equity_to_assets_rank_sector",
        "working_capital_to_assets_rank_sector",
        "debt_to_assets_low_rank_sector",
        "debt_to_equity_low_rank_sector",
        "goodwill_to_assets_low_rank_sector",
    ]
    liquidity_components = [
        "current_ratio_rank_sector",
        "quick_ratio_rank_sector",
        "working_capital_to_assets_rank_sector",
    ]
    piotroski_components = [
        "roa_rank_sector",
        "roe_rank_sector",
        "net_margin_rank_sector",
        "revenue_growth_yoy_rank_sector",
        "revenue_growth_3y_rank_sector",
        "earnings_yield_rank_sector",
        "debt_to_equity_low_rank_sector",
        "current_ratio_rank_sector",
        "quick_ratio_rank_sector",
    ]

    out["value_score"] = _mean_existing_columns(out, value_components)
    out["quality_score"] = _mean_existing_columns(out, quality_components)
    out["growth_score"] = _mean_existing_columns(out, growth_components)
    out["balance_sheet_score"] = _mean_existing_columns(out, balance_sheet_components)
    out["liquidity_score"] = _mean_existing_columns(out, liquidity_components)
    out["piotroski_style_score"] = _mean_existing_columns(out, piotroski_components)
    out["quality_value_score"] = _mean_existing_columns(out, ["quality_score", "value_score"])
    return out


def _normalize_feature_set(feature_set: FeatureSet) -> FeatureSet:
    if feature_set == "all":
        return "scores"
    if feature_set not in {"base", "ranks", "scores"}:
        raise ValueError("feature_set must be one of: base, ranks, scores, all.")
    return feature_set


def make_feature_frame(
    df: pd.DataFrame,
    fit_columns: list[str] | None = None,
    feature_set: FeatureSet = "scores",
) -> pd.DataFrame:
    """Build a numeric feature matrix without target, id, ticker, or raw dates."""
    selected_feature_set = _normalize_feature_set(feature_set)
    out = df.copy()
    out = _add_date_features(out)
    out = _add_missing_flags(out)
    out = _add_signed_log_features(out)
    out = _add_derived_ratios(out)
    if selected_feature_set in {"ranks", "scores"}:
        out = _add_rank_features(out)
    if selected_feature_set == "scores":
        out = _add_piotroski_style_features(out)
        out = _add_composite_scores(out)

    out = out.drop(columns=[col for col in FORBIDDEN_FEATURE_COLUMNS if col in out.columns], errors="ignore")
    out = out.select_dtypes(include=[np.number, "bool"]).copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    if fit_columns is not None:
        for col in fit_columns:
            if col not in out.columns:
                out[col] = np.nan
        out = out[fit_columns]
    else:
        out = out.reindex(sorted(out.columns), axis=1)

    return out.astype("float64")
