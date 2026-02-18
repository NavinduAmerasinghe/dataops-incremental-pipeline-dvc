
"""
train_model.py
- Trains baseline (lag1) and Ridge regression models on gold dataset with time-based split.
- Evaluates using metrics_model, logs to MLflow, registers best model, saves evidence.
"""
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from features_model import build_features
from data_version import get_data_version_id, get_git_commit_hash
from metrics import evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def time_split(X, y, split_ratio=0.8):
    n = len(X)
    split = int(n * split_ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def train_and_evaluate(X_train, y_train, X_val, y_val, experiment_name, data_version_id, git_hash, register, seed):
    mlflow.set_experiment(experiment_name)
    results = {}
    # Baseline: lag1 naive
    y_pred_baseline = X_val['lag1_humidity'].values
    metrics_baseline = evaluate(y_val, y_pred_baseline)
    results['baseline'] = metrics_baseline
    with mlflow.start_run(run_name="baseline_lag1"):
        mlflow.log_param("model_type", "lag1_naive")
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("data_version_id", data_version_id)
        mlflow.log_param("git_commit_hash", git_hash)
        for k, v in metrics_baseline.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(__file__)
    # Ridge regression
    model = Ridge(random_state=seed)
    model.fit(X_train, y_train)
    y_pred_ridge = model.predict(X_val)
    metrics_ridge = evaluate(y_val, y_pred_ridge)
    results['ridge'] = metrics_ridge
    with mlflow.start_run(run_name="ridge_regression") as run:
        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("data_version_id", data_version_id)
        mlflow.log_param("git_commit_hash", git_hash)
        for k, v in metrics_ridge.items():
            mlflow.log_metric(k, float(v))
        # Residual plot
        residuals = y_val - y_pred_ridge
        plt.figure(figsize=(8,4))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title("Residuals (Validation)")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plot_path = "residuals.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)
        mlflow.sklearn.log_model(model, "model")
        if register:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "climate_forecast_model")
    return results, model

def save_evidence(results, data_version_id, git_hash, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Data version: {data_version_id}\n")
        f.write(f"Git commit: {git_hash}\n")
        for model, metrics in results.items():
            f.write(f"Model: {model}\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="Path to gold.csv")
    parser.add_argument("--experiment-name", type=str, default="climate_forecast", help="MLflow experiment name")
    parser.add_argument("--register", action="store_true", help="Register best model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)
    # Load data and features
    X, y = build_features(args.data_path)
    # Time-based split
    X_train, X_val, y_train, y_val = time_split(X, y, split_ratio=0.8)
    # Data version and git hash
    data_version_id, _, _ = get_data_version_id()
    git_hash = get_git_commit_hash()
    # Train and evaluate
    results, model = train_and_evaluate(X_train, y_train, X_val, y_val, args.experiment_name, data_version_id, git_hash, args.register, args.seed)
    # Save evidence
    os.makedirs("evidence", exist_ok=True)
    save_evidence(results, data_version_id, git_hash, os.path.join("evidence", "model_summary.txt"))
    print("Training complete. Evidence saved to evidence/model_summary.txt")

if __name__ == "__main__":
    main()
