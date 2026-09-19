from maintainx.models.baseline import train_random_forest
from maintainx.models.utils import load_train_val_split
from maintainx.evaluation.metrics import evaluate_model
from maintainx.evaluation.results import log_result

X_train, y_train, X_val, y_val = load_train_val_split()
model = train_random_forest(X_train, y_train)
preds = model.predict(X_val)
log_result("Random Forest", evaluate_model(y_val, preds))