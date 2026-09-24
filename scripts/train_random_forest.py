from maintainx.models.baseline import train_random_forest
from maintainx.models.utils import load_train_val_split
from maintainx.evaluation.metrics import evaluate_model
from maintainx.evaluation.results import log_result
from maintainx.models.utils import load_test_set
from maintainx.tracking.mlflow_utils import setup_mlflow, log_regression_run

X_train, y_train, X_val, y_val = load_train_val_split()
model = train_random_forest(X_train, y_train)
preds = model.predict(X_val)
log_result("Random Forest", evaluate_model(y_val, preds))

X_test, y_test, last_mask = load_test_set()
setup_mlflow("rul-fd001")

log_regression_run(
    run_name="random_forest",
    model=model,
    params={
        "model": "RandomForest",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "window": 10,
        "seed": 42,
    },
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test, last_mask=last_mask,
)
