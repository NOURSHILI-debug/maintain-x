import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from maintainx.models.utils import load_train_val_split
from maintainx.features.pipeline import build_features
import pandas as pd
from pathlib import Path

# Load full training set (not the train/val split — CV will do its own splitting)
base_dir = Path(__file__).resolve().parents[1]
train = pd.read_parquet(base_dir / "data/processed/FD001/train.parquet")
test = pd.read_parquet(base_dir / "data/processed/FD001/test.parquet")
train_features, _ = build_features(train, test)

feature_cols = [c for c in train_features.columns if c not in ("unit_id", "cycle", "rul")]
X = train_features[feature_cols]
y = train_features["rul"]
groups = train_features["unit_id"]  # this is what makes the CV leakage-safe

param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

model = XGBRegressor(random_state=42, n_jobs=-1)

group_kfold = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=30,                      # number of random combinations to try
    scoring="neg_root_mean_squared_error",
    cv=group_kfold,
    n_jobs=-1,
    random_state=42,
    verbose=2,
)

search.fit(X, y, groups=groups)

print("Best RMSE:", -search.best_score_)
print("Best params:", search.best_params_)