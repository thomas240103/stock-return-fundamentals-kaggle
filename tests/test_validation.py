from __future__ import annotations

import pandas as pd

import pytest

from stock_returns.validation import get_time_split, percent_improvement, rmse, rmse_as_target_std_pct


def test_time_split_uses_2019_2021_train_and_2022_validation() -> None:
    df = pd.DataFrame({"start_year": [2019, 2020, 2021, 2022, 2024]})
    train_mask, validation_mask = get_time_split(df)
    assert df.loc[train_mask, "start_year"].max() < 2022
    assert set(df.loc[validation_mask, "start_year"]) == {2022}
    assert not validation_mask.iloc[4]


def test_rmse() -> None:
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx((4 / 3) ** 0.5)


def test_percent_improvement_lower_is_better() -> None:
    assert percent_improvement(100.0, 90.0) == pytest.approx(10.0)
    assert percent_improvement(100.0, 110.0) == pytest.approx(-10.0)


def test_rmse_as_target_std_pct() -> None:
    assert rmse_as_target_std_pct([0.0, 2.0], 1.0) == pytest.approx(100.0)
