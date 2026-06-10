from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.train import train_validation_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate the stock-return pipeline.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics and model artifacts.")
    parser.add_argument("--feature-set", choices=["base", "ranks", "scores", "all"], default="scores")
    parser.add_argument("--max-missing-fraction", type=float, default=0.98, help="Drop feature columns above this train missing fraction.")
    parser.add_argument("--no-optional-models", action="store_true", help="Disable optional LightGBM/XGBoost/CatBoost models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_validation_pipeline(
        train_path=args.train,
        output_dir=args.output_dir,
        include_optional=not args.no_optional_models,
        feature_set=args.feature_set,
        max_missing_fraction=args.max_missing_fraction,
    )


if __name__ == "__main__":
    main()
