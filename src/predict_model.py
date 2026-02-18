
"""
Prediction script for next-day humidity using latest MLflow model.

Steps:
    - Load latest registered MLflow model (climate_forecast_model)
    - Load data/gold/test.csv
    - Apply same feature pipeline
    - Output predictions_test.csv in project root
    - Log inference run to MLflow
    - Save inference summary to evidence/inference_log.txt
"""

import os
import mlflow
import pandas as pd
import numpy as np
from src.features_model import build_features
from src.data_version import get_data_version_id, get_git_commit_hash

def main():
    test_path = "data/gold/test.csv"
    output_path = "predictions_test.csv"
    evidence_path = "evidence/inference_log.txt"
    model_name = "climate_forecast_model"

    # Load test data and build features
    X_test, _ = build_features(csv_path=test_path)

    # Load latest registered model from MLflow Model Registry
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)

    # Output predictions
    df_pred = pd.DataFrame({"prediction": y_pred})
    df_pred.to_csv(output_path, index=False)

    # Data version and git info
    data_version_info = get_data_version_id(dataset_path=test_path)
    git_commit = get_git_commit_hash()

    # MLflow logging
    mlflow.set_experiment("inference_climate_forecast")
    with mlflow.start_run(run_name="inference_run"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_version_id", data_version_info["data_version_id"])
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("input_file", test_path)
        mlflow.log_artifact(output_path)
        # Save inference summary
        os.makedirs("evidence", exist_ok=True)
        summary = {
            "model_name": model_name,
            "data_version_id": data_version_info["data_version_id"],
            "row_count": data_version_info["row_count"],
            "dataset_path": data_version_info["dataset_path"],
            "git_commit": git_commit,
            "output_file": output_path
        }
        with open(evidence_path, "w", encoding="utf-8") as f:
            import json
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(evidence_path)

    print(f"Predictions saved to {output_path}")
    print(f"Inference summary saved to {evidence_path}")

if __name__ == "__main__":
    main()
