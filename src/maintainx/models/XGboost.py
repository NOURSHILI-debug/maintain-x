# add to src/maintainx/models/baseline.py, or a new models/xgboost_model.py — your call
from xgboost import XGBRegressor

def train_xgboost(X_train, y_train, n_estimators=200, max_depth=6, learning_rate=0.1, seed=42):
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model