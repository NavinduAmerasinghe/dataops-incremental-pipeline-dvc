
"""
Feature engineering for next-day humidity prediction using data/gold/gold.csv.

Features:
    - lag1 humidity
    - lag1 temperature (if exists)
    - rolling 7-day humidity mean
    - month
    - day of week
Target:
    - next-day humidity (shift -1)
Notes:
    - Datetime column is parsed automatically
    - Last row is dropped to avoid target leakage
    - No future data is used in features
    - Returns X (features), y (target)
"""

import pandas as pd
import os
from typing import Tuple

def build_features(
    csv_path: str = "data/gold/gold.csv",
    datetime_col: str = None,
    humidity_col: str = "humidity",
    temperature_col: str = "temperature"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build features for next-day humidity prediction.

    Args:
        csv_path: Path to the gold dataset CSV.
        datetime_col: Name of the datetime column (auto-detect if None).
        humidity_col: Name of the humidity column.
        temperature_col: Name of the temperature column (optional).
    Returns:
        X: DataFrame of features
        y: Series of next-day humidity (target)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=True)
    # DEBUG: Print columns for inspection
    print("[DEBUG] DataFrame columns:", list(df.columns))

    # --- DYNAMIC HUMIDITY COLUMN DETECTION ---
    # ModelOps best practice: Dataset schemas may change across versions, so we must
    # dynamically detect the humidity column instead of hardcoding. This ensures
    # feature engineering is robust to evolving data pipelines and supports reproducibility.
    # Add 'target' as a fallback for legacy or generic datasets
    preferred_candidates = ["humidity", "Humidity", "hum", "relative_humidity", "RH", "target"]
    humidity_col_found = None
    for candidate in preferred_candidates:
        if candidate in df.columns:
            humidity_col_found = candidate
            break
    if humidity_col_found is None:
        # Search for any column containing 'humid' (case-insensitive)
        for col in df.columns:
            if "humid" in col.lower():
                humidity_col_found = col
                break
    if humidity_col_found is None:
        raise ValueError(f"No humidity column found. Available columns: {list(df.columns)}")
    humidity_col = humidity_col_found
    print(f"[DEBUG] Using humidity column: {humidity_col}")
    # Auto-detect datetime column if not provided
    if datetime_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                break
        if datetime_col is None:
            # Try to parse first column as datetime
            try:
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="raise")
                datetime_col = df.columns[0]
            except Exception:
                raise ValueError("No datetime column found or could not parse.")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Target: next-day humidity (dynamically detected column)
    # Use shift(-1) to align each row's features with the next day's humidity value as the target
    df["target_humidity"] = df[humidity_col].shift(-1)

    # Features
    df["lag1_humidity"] = df[humidity_col].shift(1)
    if temperature_col in df.columns:
        df["lag1_temperature"] = df[temperature_col].shift(1)
    # Rolling 7-day mean (using past 7 days, no leakage)
    df["rolling7_humidity_mean"] = df[humidity_col].shift(1).rolling(window=7, min_periods=1).mean()
    # Month and day of week
    df["month"] = df[datetime_col].dt.month
    df["day_of_week"] = df[datetime_col].dt.dayofweek

    # Drop last row (target is NaN)
    df = df.iloc[:-1]

    feature_cols = ["lag1_humidity", "rolling7_humidity_mean", "month", "day_of_week"]
    if "lag1_temperature" in df.columns:
        feature_cols.append("lag1_temperature")
    X = df[feature_cols].copy()
    y = df["target_humidity"].copy()
    # Drop any rows with NaN in features or target (common after shifting/rolling)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    return X, y

if __name__ == "__main__":
    X, y = build_features()
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print(X.head())
    print(y.head())
