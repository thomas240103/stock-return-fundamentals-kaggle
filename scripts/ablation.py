from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.ablation import run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-block ablation on the 2022 validation split.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--output", default="outputs/ablation_results.csv", help="Output ablation CSV path.")
    parser.add_argument("--primary-model", default="gradient_boosting", help="Model used for feature-block comparisons.")
    parser.add_argument("--include-optional-models", action="store_true", help="Allow optional LightGBM/XGBoost/CatBoost models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_ablation(
        train_path=args.train,
        output_path=args.output,
        primary_model=args.primary_model,
        include_optional=args.include_optional_models,
    )
    print(results.to_string(index=False))
    print(f"Saved ablation results to {args.output}")


if __name__ == "__main__":
    main()
