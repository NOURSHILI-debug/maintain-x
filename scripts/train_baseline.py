import pandas as pd
from maintainx.features.pipeline import build_features
from maintainx.models.baseline import train_linear_regression, train_random_forest
from maintainx.models.utils import split_by_unit
from maintainx.evaluation.metrics import evaluate_model

# Load raw processed data
train = pd.read_parquet("data/processed/FD001/train.parquet")
test = pd.read_parquet("data/processed/FD001/test.parquet")

# Build features
train_features, test_features = build_features(train, test)
print("Feature shapes:", train_features.shape, test_features.shape)

# Split by unit_id
train_split, val_split = split_by_unit(train_features)

feature_cols = [c for c in train_split.columns if c not in ("unit_id", "cycle", "rul")]
X_train, y_train = train_split[feature_cols], train_split["rul"]
X_val, y_val = val_split[feature_cols], val_split["rul"]

lr_model = train_linear_regression(X_train, y_train)
lr_preds = lr_model.predict(X_val)
lr_metrics = evaluate_model(y_val, lr_preds)
print(f"Linear Regression — RMSE: {lr_metrics['rmse']:.2f}, "
      f"MAE: {lr_metrics['mae']:.2f}, NASA Score: {lr_metrics['nasa_score']:.2f}")

# Random Forest
rf_model = train_random_forest(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_metrics = evaluate_model(y_val, rf_preds)
print(f"Random Forest — RMSE: {rf_metrics['rmse']:.2f}, "
      f"MAE: {rf_metrics['mae']:.2f}, NASA Score: {rf_metrics['nasa_score']:.2f}")