# Modeling Plan

## Phase 0: Baselines

- Implement global mean baseline.
- Implement global median baseline.
- Implement sector median baseline.
- Establish validation RMSE on the 2022 temporal split.

## Phase 1: Sklearn-Only Robust Pipeline

- Build features from fundamentals.
- Train `HistGradientBoostingRegressor`.
- Train `ExtraTreesRegressor`.
- Train Ridge regression on rank-style features.
- Keep the project functional without optional gradient boosting libraries.

## Phase 2: Feature Engineering

- Add rank features.
- Add sector-relative features.
- Add Piotroski-style binary signals.
- Add literature-informed composite score blocks:
  - `value_score`;
  - `quality_score`;
  - `growth_score`;
  - `balance_sheet_score`;
  - `liquidity_score`;
  - `piotroski_style_score`;
  - `quality_value_score`.
- Keep score blocks as model inputs rather than hand-weighted prediction formulas.
- Add signed log transforms for skewed monetary columns.
- Use target clipping during model fitting.
- Run feature-block ablations on the 2022 validation split and save results to `outputs/ablation_results.csv`.

## Phase 3: Ensemble and Shrinkage

- Combine model predictions with validation-searched non-negative weights.
- Use neutral equal weights only as a fallback.
- Tune shrinkage alpha on the 2022 validation split.
- Save validation predictions and metrics for inspection.

## Phase 4: Optional Boosters

- Add LightGBM, XGBoost, and CatBoost only as optional dependencies.
- Keep their absence non-fatal.
- Compare their validation RMSE against the sklearn-only baseline.

## Phase 5: Leaderboard Discipline

- Use the public leaderboard sparingly.
- Submit only a few informative variants.
- Treat validation RMSE and model diagnostics as the primary feedback loop.
- Avoid changing weights or hyperparameters only because of one leaderboard result.
