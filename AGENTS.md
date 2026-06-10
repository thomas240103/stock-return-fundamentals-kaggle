# Agent Instructions

Before editing code, read this file completely.
Before changing modeling logic, read:
- README.md
- docs/modeling_plan.md
- docs/kaggle_rules_summary.md
- docs/research_notes.md

## Permanent Rules

- Do not modify the competition objective: predict 1-year `return_pct`.
- Do not use leakage.
- Do not use the test set to choose features, hyperparameters, or ensemble weights.
- Do not commit Kaggle data.
- Do not add external data without documenting it and making it optional.
- Every script must be reproducible from the CLI.
- Every important model change must update `docs/modeling_plan.md`.
- Every new feature must be documented in `features.py` and, if important, in the README.
- Every submission must be saved in `outputs/` with a timestamp or descriptive name when it is not the canonical `outputs/submission.csv`.
- Every new dependency must be open source and compatible with this project.
- Prefer robust models and temporal validation over leaderboard overfitting.
- Keep the code readable and modular.

## Required Tests

Write or update tests for:

- submission format;
- absence of the target in the feature matrix;
- correct temporal split;
- absence of forbidden columns;
- correct test output shape.

## Data Rules

The repository must not contain:

- `train.csv`;
- `test.csv`;
- `sample_submission.csv`;
- raw Kaggle downloads;
- generated model artifacts in `outputs/`.

Keep placeholder `.gitkeep` files in data and output folders so the directory layout remains visible.

## Modeling Rules

Use `start_year < 2022` for training and `start_year == 2022` for validation unless `docs/modeling_plan.md` is deliberately updated with a stronger temporal validation design.

Fit models on clipped targets when configured, but always evaluate validation RMSE against the real, unclipped target.

Keep the final submission columns exactly:

```csv
id,return_pct
```
