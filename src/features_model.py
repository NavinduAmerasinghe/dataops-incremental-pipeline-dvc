
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

    # Target: next-day humidity
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
    return X, y

if __name__ == "__main__":
    X, y = build_features()
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print(X.head())
    print(y.head())
