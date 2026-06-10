# Research Notes

## Forecast Combination

Bates and Granger (1969) show that combining forecasts can reduce error when individual models capture different parts of the signal or make partially independent mistakes. This matters here because fundamentals can be weak predictors of stock returns; a single model may overreact to noise, while a combination can be more stable.

## Stacking

Wolpert (1992) introduced stacked generalization, where a meta-model learns how to combine base model predictions. In this project, stacking is treated cautiously because there is only a limited temporal validation window. The implementation favors a small validation-based non-negative weight search with neutral fallback over aggressive meta-learning.

## Empirical Asset Pricing With Machine Learning

Gu, Kelly, and Xiu (2020) show that machine learning methods can help empirical asset pricing by capturing nonlinearities and interactions across firm characteristics. Tree models are useful for interactions such as valuation ratios behaving differently by profitability, growth, leverage, or sector.

## Fundamental Analysis and Stock Prediction

Huang, Capretz, and Ho (2022) discuss machine learning for stock prediction using fundamental analysis, including quarterly financial data, Random Forest models, feature selection, and aggregated forecasts. This repository follows that practical direction while using temporal validation to reduce look-ahead risk.

## Piotroski-Style Accounting Signals

Piotroski (2000) uses simple accounting indicators to distinguish stronger and weaker firms among value stocks. This project includes binary signals such as positive ROA, positive ROE, positive margins, positive growth, and sector-relative liquidity/leverage features. It also builds a rank-based `piotroski_style_score` so the signal can be used as a model input without replacing the ML model with a fixed accounting formula.

## Accruals / Earnings Quality

Sloan (1996) highlights that accruals and cash-flow related information can carry predictive content. The current dataset does not directly expose all accrual components, but balance-sheet and income-statement ratios can still provide partial quality and sustainability signals.

## Why Ranks and Sector Normalization Help

Raw fundamentals are not always comparable across sectors. Banks, utilities, software companies, and industrial firms can have very different balance sheets and valuation norms. Global ranks provide scale-robust ordering, while sector ranks and sector z-scores compare firms against closer peers.

The composite score blocks follow the same idea. `value_score`, `quality_score`, `growth_score`, `balance_sheet_score`, `liquidity_score`, `piotroski_style_score`, and `quality_value_score` summarize related sector-relative percentile ranks. They are compact features for the model, not final prediction formulas.

## Why Target Clipping and Shrinkage Help Under RMSE

RMSE punishes large errors heavily. Stock returns are fat-tailed, so fitting directly to extreme outcomes can make models unstable. Clipping the training target between the 1st and 99th percentiles reduces the influence of extreme observations during fitting. Shrinkage toward the training mean reduces prediction variance and can improve RMSE when signal is weak.

## Why Temporal Validation Matters

Random splits can place nearby observations, market regimes, or repeated tickers across both train and validation. That can produce optimistic validation scores. The competition has a time gap between training and test, so validation should respect time. This project uses 2019-2021 for training and 2022 for validation.
