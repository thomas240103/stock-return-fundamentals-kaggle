from __future__ import annotations

import numpy as np
import pandas as pd

from stock_returns.features import COMPOSITE_SCORE_COLUMNS, make_feature_frame


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ticker": ["AAA", "BBB", "CCC"],
            "start_year": [2019, 2020, 2022],
            "period_start": ["2019-01-01", "2020-04-01", None],
            "period_end": ["2019-03-31", "2020-06-30", "2022-12-31"],
            "return_pct": [10.0, -5.0, 20.0],
            "market_cap": [1000.0, 0.0, np.nan],
            "revenue_ttm": [100.0, 0.0, -50.0],
            "net_income_ttm": [10.0, -1.0, np.nan],
            "long_term_debt": [20.0, 0.0, 5.0],
            "total_assets": [200.0, 0.0, np.nan],
            "stockholders_equity": [80.0, 0.0, -10.0],
            "current_assets": [50.0, 0.0, 3.0],
            "current_liabilities": [20.0, 0.0, 0.0],
            "goodwill": [5.0, np.nan, 1.0],
            "inventory": [8.0, 0.0, 2.0],
            "roe": [0.1, -0.2, np.nan],
            "roa": [0.05, 0.0, -0.1],
            "net_margin": [0.1, -0.1, np.nan],
            "gross_margin": [0.4, np.nan, 0.2],
            "operating_margin": [0.2, -0.1, 0.0],
            "revenue_growth_yoy": [0.1, -0.2, 0.0],
            "revenue_growth_3y": [0.3, np.nan, -0.1],
            "pe_ttm": [15.0, 0.0, 30.0],
            "price_to_book": [2.0, np.nan, 1.0],
            "price_to_sales": [3.0, 0.5, np.nan],
            "debt_to_equity": [0.2, 0.0, np.nan],
            "current_ratio": [2.0, 0.0, np.nan],
            "quick_ratio": [1.5, 0.0, np.nan],
            "dividend_yield": [0.01, 0.0, np.nan],
            "income_before_tax": [9.0, -2.0, np.nan],
            "shares_outstanding": [100.0, 200.0, np.nan],
            "shares_diluted": [110.0, 210.0, np.nan],
            "sector_code": [0, 0, 1],
        }
    )


def test_feature_frame_excludes_target_and_forbidden_columns() -> None:
    features = make_feature_frame(_sample_frame())
    assert "return_pct" not in features.columns
    assert "id" not in features.columns
    assert "ticker" not in features.columns
    assert "period_start" not in features.columns
    assert "period_end" not in features.columns


def test_feature_frame_handles_missing_and_zero_division() -> None:
    features = make_feature_frame(_sample_frame())
    assert len(features) == 3
    assert np.isfinite(features.fillna(0).to_numpy()).all()
    assert "net_income_to_revenue" in features.columns
    assert "quality_value_score" in features.columns
    for col in COMPOSITE_SCORE_COLUMNS:
        assert col in features.columns


def test_feature_frame_alignment_keeps_shape() -> None:
    train_features = make_feature_frame(_sample_frame())
    test_features = make_feature_frame(_sample_frame().drop(columns=["price_to_book"]), fit_columns=list(train_features.columns))
    assert list(test_features.columns) == list(train_features.columns)
    assert len(test_features) == len(train_features)


def test_feature_sets_control_rank_and_score_blocks() -> None:
    base_features = make_feature_frame(_sample_frame(), feature_set="base")
    rank_features = make_feature_frame(_sample_frame(), feature_set="ranks")
    score_features = make_feature_frame(_sample_frame(), feature_set="scores")

    assert "roe_rank_sector" not in base_features.columns
    assert "value_score" not in base_features.columns
    assert "roe_rank_sector" in rank_features.columns
    assert "value_score" not in rank_features.columns
    assert "roe_rank_sector" in score_features.columns
    assert "value_score" in score_features.columns
    assert len(base_features) == len(rank_features) == len(score_features)
