from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.data import load_test
from stock_returns.predict import load_bundle, predict_with_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with an existing model bundle.")
    parser.add_argument("--model", default="outputs/models/model_bundle.joblib", help="Path to a fitted model bundle.")
    parser.add_argument("--test", default="data/raw/test.csv", help="Path to Kaggle test.csv.")
    parser.add_argument("--output", default="outputs/test_predictions.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model)
    test_df = load_test(args.test)
    prediction = predict_with_bundle(bundle, test_df)
    output = pd.DataFrame({"id": test_df["id"], "return_pct": prediction})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    print(f"Saved predictions to {target}")


if __name__ == "__main__":
    main()
