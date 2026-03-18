# ─────────────────────────────────────────────
# 4. LIGHTGBM SETUP
# ─────────────────────────────────────────────

# pip install lightgbm pandas scikit-learn

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import joblib


FEATURE_COLS = [
    "diff_adj_em", "diff_adjoe", "diff_adjde", "diff_barthag",
    "diff_efg_o", "diff_efg_d", "diff_tor", "diff_tord",
    "diff_orb", "diff_drb", "diff_ftr", "diff_adj_t", "diff_wab",
    "seed_diff",
]

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

