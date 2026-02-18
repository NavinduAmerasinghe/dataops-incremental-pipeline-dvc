
"""
Metrics module for model evaluation.

Primary KPI:
    - MAE (Mean Absolute Error)
Secondary KPIs:
    - RMSE (Root Mean Squared Error)
    - Stability metric = MAE_last_30_days / MAE_overall
    - Latency (ms)
"""

import numpy as np
import time
from typing import Dict

def evaluate(y_true, y_pred, dates=None) -> Dict[str, float]:
    """
    Evaluate predictions with MAE, RMSE, stability, and latency.

    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        dates: Optional array-like of datetime (for stability metric)
    Returns:
        Dictionary of metrics
    """
    start = time.time()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Stability metric: last 30 days MAE / overall MAE
    if dates is not None:
        import pandas as pd
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "date": pd.to_datetime(dates)})
        last_30 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=29)]
        if len(last_30) > 0:
            mae_last_30 = np.mean(np.abs(last_30["y_true"] - last_30["y_pred"]))
            stability = mae_last_30 / mae if mae != 0 else np.nan
        else:
            stability = np.nan
    else:
        stability = np.nan
    latency_ms = (time.time() - start) * 1000
    return {
        "mae": mae,
        "rmse": rmse,
        "stability": stability,
        "latency_ms": latency_ms
    }

# Small test example
if __name__ == "__main__":
    y_true = [1, 2, 3, 4, 5, 6, 7]
    y_pred = [1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2]
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=7)
    metrics = evaluate(y_true, y_pred, dates)
    print("Evaluation metrics:", metrics)
