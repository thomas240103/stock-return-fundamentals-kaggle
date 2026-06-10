# Predict 1-Year US Stock Returns from Fundamentals

Open source, reproducible baseline and modeling pipeline for the Kaggle competition **Predict 1-Year US Stock Returns from Fundamentals**.

Competition URL: https://www.kaggle.com/competitions/predict-1-year-us-stock-returns-from-fundamentals

Kaggle citation:

> alexandrefrank. (2026). Predict 1-Year US Stock Returns from Fundamentals. Kaggle. Data Source: Atlantis Data Solutions. https://www.kaggle.com/competitions/predict-1-year-us-stock-returns-from-fundamentals

This repository is designed for three uses:

1. participating in the Kaggle competition;
2. making the approach readable for open source contributors;
3. allowing future AI agents to continue the work by first reading `AGENTS.md`.

## Dataset

The competition provides:

- `train.csv`: 23,070 stock-quarter observations from 2019-2022, including `return_pct`.
- `test.csv`: 8,520 observations from 2024, excluding `return_pct`.
- `sample_submission.csv`: required submission format.
- Target: `return_pct`, the 1-year percentage return.
- Metric: RMSE on `return_pct`; lower is better.

Required submission format:

```csv
id,return_pct
0,12.5
1,-3.2
2,45.0
```

The final `outputs/submission.csv` must contain a header and 8,520 prediction rows when generated against the official test set.

## Why This Problem Is Difficult

Stock returns are noisy and fat-tailed. Company fundamentals usually explain only a small part of cross-sectional return variation, and the target can contain extreme positive and negative outcomes. A model that chases rare moonshots can worsen RMSE by making large errors elsewhere. The practical goal is not to identify every extreme winner; it is to reduce large errors while preserving useful signal from valuation, profitability, growth, leverage, and sector-relative fundamentals.

There is also a deliberate time gap between training and test data. This makes temporal validation essential and discourages shortcuts that accidentally rely on future information.

## Literature

- Bates, J. M., & Granger, C. W. J. (1969). "The Combination of Forecasts." Combining forecasts can reduce error compared with selecting one model.
- Wolpert, D. H. (1992). "Stacked Generalization." Stacking and meta-learning can combine models with complementary errors.
- Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." Tree models and neural networks can capture nonlinearities and interactions in asset pricing predictors.
- Huang, Y., Capretz, L. F., & Ho, D. (2022). "Machine Learning for Stock Prediction Based on Fundamental Analysis." Quarterly financial data, Random Forests, feature selection, and forecast aggregation can be useful in fundamental analysis pipelines.
- Piotroski, J. D. (2000). "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers." Simple accounting signals and composite scores can separate stronger and weaker firms.
- Sloan, R. G. (1996). "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?" Earnings quality, accruals, and cash-flow information may have predictive content.
- Nti, I. K., Adekoya, A. F., & Weyori, B. A. (2020). "A comprehensive evaluation of ensemble learning for stock-market prediction." Reviews bagging, boosting, blending, stacking, and other ensembles for stock-market prediction.

## Practical Strategy

The pipeline uses:

- robust mean, median, and sector baselines;
- financial feature engineering;
- signed log transforms for skewed monetary variables;
- missing-value indicators;
- global ranks and sector-relative ranks;
- literature-informed composite score blocks based mostly on sector-relative percentile ranks;
- Piotroski-style binary accounting signals;
- target clipping / winsorization during model fitting;
- multiple models;
- validation-tuned ensembling;
- shrinkage toward the training mean;
- temporal validation with 2019-2021 train and 2022 validation.

## Leakage Prevention

This repository follows these rules:

- do not use `return_pct` outside supervised training;
- do not use future information;
- do not use the test set to choose features, hyperparameters, or ensemble weights;
- do not use public leaderboard feedback as the main validation method;
- do not perform target encoding across train and validation/test without temporal cross-validation;
- keep external data disabled by default.

## Installation

```bash
pip install -r requirements.txt
```

Optional boosters can be installed separately if desired:

```bash
pip install lightgbm xgboost catboost
```

The code works without these optional packages by falling back to scikit-learn models.

## Data Setup

Do not commit Kaggle data to this repository.

Download the competition files from Kaggle and place them here:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

The `.gitignore` is configured to keep raw and processed data out of git while preserving empty folders with `.gitkeep`.

## Train

```bash
python scripts/train.py --train data/raw/train.csv --output-dir outputs
```

This command:

- reads the training file;
- creates features;
- uses `start_year < 2022` for training and `start_year == 2022` for validation;
- fits baselines and models on clipped training targets;
- evaluates RMSE on the real, unclipped 2022 target;
- saves validation predictions, metrics, and model artifacts in `outputs/`.

## Feature Ablation

```bash
python scripts/ablation.py --train data/raw/train.csv --output outputs/ablation_results.csv
```

The ablation compares:

- base features only;
- base + ranks;
- base + ranks + composite scores;
- base + ranks + composite scores + validation-tuned ensemble.

Results are saved to `outputs/ablation_results.csv` with validation RMSE and improvement versus the previous stage.

## Generate Submission

```bash
python scripts/make_submission.py --train data/raw/train.csv --test data/raw/test.csv --output outputs/submission.csv
```

This command tunes ensemble/shrinkage on the temporal split when possible, refits models on all available training rows, predicts the official test rows, clips final predictions to a conservative range, and writes:

```text
outputs/submission.csv
```

## Repository Structure

```text
stock-return-fundamentals-kaggle/
├── README.md
├── AGENTS.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── docs/
│   ├── research_notes.md
│   ├── kaggle_rules_summary.md
│   └── modeling_plan.md
├── src/
│   └── stock_returns/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── validation.py
│       ├── models.py
│       ├── ensemble.py
│       ├── train.py
│       ├── predict.py
│       └── utils.py
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── make_submission.py
├── notebooks/
│   └── 01_eda.ipynb
├── tests/
│   ├── test_features.py
│   ├── test_validation.py
│   └── test_submission_format.py
└── outputs/
    └── .gitkeep
```

## Methodology

Feature engineering is implemented in `src/stock_returns/features.py`. It includes date features, signed log transforms, missing flags, derived ratios, sector-relative signals, rank features, cheapness ranks for valuation ratios, and composite score blocks:

- `value_score`;
- `quality_score`;
- `growth_score`;
- `balance_sheet_score`;
- `liquidity_score`;
- `piotroski_style_score`;
- `quality_value_score`.

These scores are additional model inputs. They do not replace the ML models with hand-weighted prediction formulas.

Models are implemented in `src/stock_returns/models.py`:

- global mean baseline;
- global median baseline;
- sector median baseline;
- `HistGradientBoostingRegressor`;
- `ExtraTreesRegressor`;
- Ridge regression on rank-style features;
- optional LightGBM, XGBoost, and CatBoost models when installed.

The ensemble is implemented in `src/stock_returns/ensemble.py`. Candidate model weights are selected by a small non-negative grid search on the 2022 validation split, with neutral equal weights as fallback. Then predictions are shrunk toward the training mean:

```python
final_pred = alpha * model_pred + (1 - alpha) * train_mean
```

`alpha` is selected on the 2022 validation split from `np.linspace(0.0, 1.0, 51)`.

## Validation

The main validation is temporal:

- train: `start_year < 2022`;
- validation: `start_year == 2022`;
- final test: 2024.

RMSE is computed on the real validation target, not the clipped target.

## Kaggle Notebook

The notebook in `notebooks/01_eda.ipynb` is intentionally thin. It loads data, runs basic EDA, creates features, and can call the project pipeline. Keep reusable logic in `src/` and CLI scripts rather than notebook-only cells.

## Limitations

- Fundamentals alone may have weak predictive power for 1-year stock returns.
- A single 2022 validation split may not fully represent 2024 behavior.
- Rank and sector features can help normalize firm differences but cannot remove macro regime shifts.
- Final clipping and shrinkage reduce RMSE risk but may underpredict extreme winners.

## Future Work

- Add time-aware cross-validation over additional historical windows if more data becomes available.
- Add careful model diagnostics by sector and market-cap regime.
- Add optional open, documented external data only if allowed and disabled by default.
- Explore constrained stacking with strong regularization.
- Track submissions and validation experiments in a lightweight experiment log.

## License

This code is released under the MIT License.

## Disclaimer

This project is for research and competition purposes only. It is not financial advice.

## Nota Breve in Italiano

Questa repo prepara una pipeline Kaggle riproducibile senza includere dati grezzi. Scarica i CSV dalla pagina della competizione, mettili in `data/raw/`, esegui `scripts/train.py` per validare e `scripts/make_submission.py` per creare `outputs/submission.csv`.
