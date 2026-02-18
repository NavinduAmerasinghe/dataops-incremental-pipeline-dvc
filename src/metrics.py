import numpy as np
import time
from datetime import datetime, timedelta

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def stability_metric(mae_history, days=30):
    if not mae_history:
        return None
    now = datetime.now()
    cutoff = now - timedelta(days=days)
    last_30 = [v for d, v in mae_history if d >= cutoff]
    all_mae = [v for d, v in mae_history]
    if not all_mae or not last_30:
        return None
    return np.mean(last_30) / np.mean(all_mae)

def evaluate(y_true, y_pred, mae_history=None, latency_ms=None):
    """
    Evaluate metrics for predictions.
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        mae_history: List of (datetime, mae_value) for stability metric
        latency_ms: Optional latency in milliseconds
    Returns:
        dict with MAE, RMSE, Stability, Latency
    """
    t0 = time.time()
    result = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
    }
    if mae_history is not None:
        result['stability'] = stability_metric(mae_history)
    if latency_ms is not None:
        result['latency_ms'] = latency_ms
    else:
        result['latency_ms'] = (time.time() - t0) * 1000
    return result

if __name__ == "__main__":
    # Small test example
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    mae_history = [
        (datetime.now() - timedelta(days=i), 0.5 + 0.01 * i) for i in range(40)
    ]
    metrics = evaluate(y_true, y_pred, mae_history=mae_history)
    print("Metrics:", metrics)
