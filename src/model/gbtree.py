# Note: LightGBM handles missing values natively — no scaling or
# imputation needed, unlike neural nets. Just pass the raw DataFrame.

LGBM_PARAMS = {
    "objective":      "binary",        # binary classification
    "metric":         ["binary_logloss", "binary_error"],
    "boosting_type":  "gbdt",
    "learning_rate":  0.05,
    "num_leaves":     31,              # keep small — limited training data
    "min_child_samples": 10,           # avoid overfitting on small tourney dataset
    "feature_fraction": 0.8,          # subsample features per tree
    "bagging_fraction": 0.8,          # subsample rows per tree
    "bagging_freq":   5,
    "lambda_l1":      0.1,            # L1 regularization
    "lambda_l2":      0.1,            # L2 regularization
    "verbose":        -1,
}

