import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Training script for next-day humidity prediction.

Features:
    - Loads gold dataset
    - Calls feature engineering
    - Time-based split (80% train, 20% validation)
    - Baseline (lag1 naive) and Ridge regression
    - Evaluates using metrics_model
    - MLflow: log params, metrics, data_version_id, git commit, residual plot
    - Register best model as 'climate_forecast_model' if --register
    - Save evidence summary to evidence/model_summary.txt
    - CLI: --data-path, --experiment-name, --register
    - Reproducibility via random seed
"""

import argparse
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
import os
import numpy as np
import pandas as pd
import mlflow
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from src.features_model import build_features
from src.metrics_model import evaluate
from src.data_version import get_data_version_id, get_git_commit_hash

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def time_split(X, y, split_ratio=0.8):
    n = len(X)
    split = int(n * split_ratio)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    return X_train, X_val, y_train, y_val

def plot_residuals(y_true, y_pred, out_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,4))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title("Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/gold/gold.csv", help="Path to gold dataset")
    parser.add_argument("--experiment-name", type=str, default="climate_forecast", help="MLflow experiment name")
    parser.add_argument("--register", action="store_true", help="Register best model to MLflow Model Registry")
    args = parser.parse_args()


    # Feature engineering
    X, y = build_features(csv_path=args.data_path)
    # Fail fast if no data
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("[ERROR] No training data after feature engineering.\n"
              f"Gold dataset path: {args.data_path}\n"
              f"Columns: {list(X.columns) if hasattr(X, 'columns') else X}\n"
              f"Shape: {X.shape}\n"
              "Check that the gold dataset is generated and contains valid data with the expected schema.\n"
              "This often means the data pipeline did not produce any usable rows.\n"
              "You can debug by printing the head of the gold file in your workflow before training.")
        import sys
        sys.exit(1)
    # Time-based split
    X_train, X_val, y_train, y_val = time_split(X, y, split_ratio=0.8)

    # Baseline: lag1 naive
    y_pred_baseline = X_val["lag1_humidity"].values
    metrics_baseline = evaluate(y_val, y_pred_baseline)


    # Hyperparameter tuning for Ridge regression (manual grid search)
    best_alpha = None
    best_score = float('inf')
    best_model = None
    best_pred = None
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    tuning_results = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        tuning_results.append({"alpha": alpha, "mae": mae, "rmse": rmse})
        mlflow.log_metric(f"ridge_mae_alpha_{alpha}", mae)
        mlflow.log_metric(f"ridge_rmse_alpha_{alpha}", rmse)
        if mae < best_score:
            best_score = mae
            best_alpha = alpha
            best_model = model
            best_pred = y_pred
    # Log best alpha
    mlflow.log_param("ridge_best_alpha", best_alpha)
    # Log all tuning results as artifact
    import json
    os.makedirs("evidence", exist_ok=True)
    with open("evidence/ridge_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(tuning_results, f, indent=2)
    mlflow.log_artifact("evidence/ridge_tuning_results.json")
    # Use best model
    model = best_model
    y_pred_ridge = best_pred
    metrics_ridge = evaluate(y_val, y_pred_ridge)

    # Data version and git info
    data_version_info = get_data_version_id(dataset_path=args.data_path)
    git_commit = get_git_commit_hash()

    # MLflow logging
    mlflow.set_experiment(args.experiment_name)
    # End any previous run before starting a new one
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("data_version_id", data_version_info["data_version_id"])
        mlflow.log_param("git_commit", git_commit)
        for k, v in metrics_ridge.items():
            mlflow.log_metric(f"ridge_{k}", v)
        for k, v in metrics_baseline.items():
            mlflow.log_metric(f"baseline_{k}", v)
        # Residual plot
        os.makedirs("evidence", exist_ok=True)
        residual_plot_path = "evidence/residuals.png"
        plot_residuals(y_val, y_pred_ridge, residual_plot_path)
        mlflow.log_artifact(residual_plot_path)
        # Save model
        model_path = "evidence/ridge_model.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "model")
        # Register model if requested
        if args.register:
            result = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name="climate_forecast_model"
            )
            print(f"Model registered: {result.name} v{result.version}")
        # Evidence summary
        summary = {
            "data_version_id": data_version_info["data_version_id"],
            "row_count": data_version_info["row_count"],
            "dataset_path": data_version_info["dataset_path"],
            "git_commit": git_commit,
            "metrics_ridge": metrics_ridge,
            "metrics_baseline": metrics_baseline
        }
        with open("evidence/model_summary.txt", "w", encoding="utf-8") as f:
            import json
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("evidence/model_summary.txt")

    print("Training complete. Metrics:")
    print("Ridge:", metrics_ridge)
    print("Baseline:", metrics_baseline)

if __name__ == "__main__":
    main()
