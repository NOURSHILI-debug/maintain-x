import mlflow
from maintainx.tracking.mlflow_utils import setup_mlflow, tag_run

uri = setup_mlflow("smoke-test")
print("Tracking URI:", uri)

with mlflow.start_run(run_name="env-test"):
    tag_run()
    mlflow.log_param("source", ".env")
    mlflow.log_metric("rmse", 1.23)
    with open("env_test.txt", "w") as f:
        f.write("hi")
    mlflow.log_artifact("env_test.txt")

print("Done")