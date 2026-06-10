# Kaggle Rules Summary

This is an operational summary for contributors. It is not legal advice and does not replace the official Kaggle competition rules.

## Competition

- Title: Predict 1-Year US Stock Returns from Fundamentals.
- Sponsor: Atlantis Data Solutions.
- Sponsor address listed in the rules: 1606 Headway Cir STE 9703, Austin, TX 78754, US.
- Prize: $1000 total, first prize $1000.
- Winner license: Open Source. Winning source code must use an OSI-approved license that does not limit commercial use.
- Data source: features are derived from SEC filings, including 10-K and 10-Q filings.
- Evaluation metric: RMSE on `return_pct`; lower is better.
- Official submission format: `id,return_pct` with exactly 8,520 prediction rows for the official test set.

## Practical Constraints

- Competition data use is for non-commercial and academic research purposes under the competition rules.
- Team size is limited to a maximum of 5 members.
- Maximum daily submissions: 5.
- Maximum final submissions: 2.
- External data must be reasonably accessible, documented, and allowed by the official rules.
- Do not privately share competition code or data outside the team.
- During the competition, public code sharing should happen through Kaggle forums or Kaggle notebooks if the official rules require that channel.
- Winners may need to provide complete code, training code, inference code, environment details, and documentation.
- Winner documentation may need to explain methodology, preprocessing, loss function, training details, hyperparameters, environment, and how to reproduce the winning submission.
- Do not commit competition data to git.

## Repository Rule

This repository tracks code, documentation, tests, and empty folder placeholders only. Raw Kaggle CSV files must stay in `data/raw/` locally and remain untracked.
