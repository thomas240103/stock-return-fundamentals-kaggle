from __future__ import annotations

import pandas as pd
import pytest

from stock_returns.predict import validate_submission_frame


def test_submission_format_accepts_valid_submission() -> None:
    test_df = pd.DataFrame({"id": [0, 1, 2], "feature": [10, 20, 30]})
    submission = pd.DataFrame({"id": [0, 1, 2], "return_pct": [1.0, -2.0, 3.0]})
    validate_submission_frame(submission, test_df, expected_rows=3)


def test_submission_format_rejects_wrong_rows() -> None:
    test_df = pd.DataFrame({"id": [0, 1, 2]})
    submission = pd.DataFrame({"id": [0, 1], "return_pct": [1.0, 2.0]})
    with pytest.raises(ValueError):
        validate_submission_frame(submission, test_df)


def test_submission_format_rejects_nan_predictions() -> None:
    test_df = pd.DataFrame({"id": [0, 1]})
    submission = pd.DataFrame({"id": [0, 1], "return_pct": [1.0, None]})
    with pytest.raises(ValueError):
        validate_submission_frame(submission, test_df)


def test_submission_format_rejects_wrong_official_row_count() -> None:
    test_df = pd.DataFrame({"id": [0, 1, 2]})
    submission = pd.DataFrame({"id": [0, 1, 2], "return_pct": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="exactly 8520 rows"):
        validate_submission_frame(submission, test_df, expected_rows=8520)
