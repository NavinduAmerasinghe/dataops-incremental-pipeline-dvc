"""
Script to collect model evidence for ModelOps:
- Writes model_versions.csv from MLflow runs
- Writes current data_version_id
- Saves metrics summary
- Creates reproducibility instructions text file
"""

import os
import mlflow
import pandas as pd
from src.data_version import get_data_version_id

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# 1. Write model_versions.csv from MLflow runs
def write_model_versions(experiment_name="climate_forecast", out_path=f"{EVIDENCE_DIR}/model_versions.csv"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return
    runs = client.search_runs(experiment.experiment_id, "", order_by=["attributes.start_time DESC"])
    records = []
    for run in runs:
        d = run.data.metrics.copy()
        d.update(run.data.params)
        d["run_id"] = run.info.run_id
        d["start_time"] = run.info.start_time
        d["end_time"] = run.info.end_time
        records.append(d)
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"Model versions written to {out_path}")

# 2. Write current data_version_id
def write_data_version(out_path=f"{EVIDENCE_DIR}/data_version_id.txt", data_path="data/gold/gold.csv"):
    info = get_data_version_id(dataset_path=data_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(info["data_version_id"] + "\n")
    print(f"Data version ID written to {out_path}")

# 3. Save metrics summary (from latest run)
def write_metrics_summary(experiment_name="climate_forecast", out_path=f"{EVIDENCE_DIR}/metrics_summary.json"):
    import json
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return
    runs = client.search_runs(experiment.experiment_id, "", order_by=["attributes.start_time DESC"])
    if not runs:
        print("No runs found.")
        return
    latest = runs[0]
    summary = {"metrics": latest.data.metrics, "params": latest.data.params, "run_id": latest.info.run_id}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary written to {out_path}")

# 4. Create reproducibility instructions
def write_repro_instructions(out_path=f"{EVIDENCE_DIR}/repro_instructions.txt"):
    text = (
        "Reproducibility Instructions:\n"
        "1. Clone the repository and install dependencies from requirements.txt.\n"
        "2. Ensure DVC and MLflow are installed and configured.\n"
        "3. Use the same data_version_id as in data_version_id.txt.\n"
        "4. Run: python src/train_model.py --register\n"
        "5. Run: python src/predict_model.py\n"
        "6. Compare metrics and predictions with those in evidence/.\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Reproducibility instructions written to {out_path}")

if __name__ == "__main__":
    write_model_versions()
    write_data_version()
    write_metrics_summary()
    write_repro_instructions()
