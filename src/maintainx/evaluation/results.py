import json
from pathlib import Path
from datetime import datetime

RESULTS_PATH = Path(__file__).resolve().parents[3] / "scripts" / "results.json"

def log_result(model_name, metrics):
    """Append a model's evaluation results to a shared results file."""
    results = []
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())

    results.append({
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    })
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"{model_name} — " + ", ".join(f"{k}: {v:.2f}" for k, v in metrics.items()))