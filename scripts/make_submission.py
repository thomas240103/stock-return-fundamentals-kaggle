from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.predict import make_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Kaggle submission.csv.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--test", default="data/raw/test.csv", help="Path to Kaggle test.csv.")
    parser.add_argument("--output", default="outputs/submission.csv", help="Output submission CSV path.")
    parser.add_argument("--model-output", default="outputs/models/full_train_bundle.joblib", help="Optional fitted model bundle output.")
    parser.add_argument("--feature-set", choices=["base", "ranks", "scores", "all"], default="scores")
    parser.add_argument("--max-missing-fraction", type=float, default=0.98, help="Drop feature columns above this train missing fraction.")
    parser.add_argument("--no-optional-models", action="store_true", help="Disable optional LightGBM/XGBoost/CatBoost models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = make_submission(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        include_optional=not args.no_optional_models,
        model_output_path=args.model_output,
        feature_set=args.feature_set,
        max_missing_fraction=args.max_missing_fraction,
    )
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
