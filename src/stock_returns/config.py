"""Project constants."""

from __future__ import annotations

TARGET_COL = "return_pct"
ID_COL = "id"
SECTOR_COL = "sector_code"
EXPECTED_TEST_ROWS = 8520
COMPETITION_FEATURE_COUNT = 39
RANDOM_STATE = 42
EPSILON = 1e-9

FORBIDDEN_FEATURE_COLUMNS = {
    TARGET_COL,
    ID_COL,
    "ticker",
    "period_start",
    "period_end",
}

DATE_COLUMNS = ["period_start", "period_end"]

SIGNED_LOG_COLUMNS = [
    "market_cap",
    "revenue_ttm",
    "net_income_ttm",
    "income_before_tax",
    "total_assets",
    "stockholders_equity",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "goodwill",
    "inventory",
    "dividends_paid_ttm",
    "shares_outstanding",
    "shares_diluted",
]

RANK_COLUMNS = [
    "roe",
    "roa",
    "net_margin",
    "gross_margin",
    "operating_margin",
    "revenue_growth_yoy",
    "revenue_growth_3y",
    "pe_ttm",
    "price_to_book",
    "price_to_sales",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "dividend_yield",
    "earnings_yield",
    "sales_yield",
    "net_income_to_revenue",
    "debt_to_assets",
    "equity_to_assets",
    "working_capital_to_assets",
    "goodwill_to_assets",
    "inventory_to_assets",
]

VALUATION_COLUMNS = ["pe_ttm", "price_to_book", "price_to_sales"]

LOWER_IS_BETTER_COLUMNS = [
    "pe_ttm",
    "price_to_book",
    "price_to_sales",
    "debt_to_equity",
    "debt_to_assets",
    "goodwill_to_assets",
    "inventory_to_assets",
]

SECTOR_MAPPING = {
    0: "Technology",
    1: "Financial Services",
    2: "Industrials",
    3: "Healthcare",
    4: "Consumer Cyclical",
    5: "Real Estate",
    6: "Basic Materials",
    7: "Consumer Defensive",
    8: "Energy",
    9: "Utilities",
    10: "Communication Services",
}

PREFERRED_ENSEMBLE_MODELS = ["gradient_boosting", "extra_trees", "ridge_rank", "sector_median"]
DEFAULT_ENSEMBLE_WEIGHTS = {name: 1.0 for name in PREFERRED_ENSEMBLE_MODELS}

FINAL_PREDICTION_CLIP = (-100.0, 300.0)
DEFAULT_MAX_MISSING_FRACTION = 0.98
