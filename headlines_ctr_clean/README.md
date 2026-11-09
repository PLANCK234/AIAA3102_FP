Use your SuperAgent to predict which of two headlines had the higher observed click-through rate (clicks divided by impressions) in an A/B test. Each CSV contains three columns. Use `headline_A` and `headline_B` as input. The target `label_pairwise` is `1` if A had the higher rate and `0` if B did.

Use `train.csv` (8,301 pairs) and `val.csv` (940 pairs) for model development and validation. Use the labeled `test.csv` (3,232 pairs) only for final evaluation and report **test accuracy**.
