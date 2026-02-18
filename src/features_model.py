"""
Feature engineering for next-day humidity prediction using gold dataset.

Features:
- lag1 humidity
- lag1 temperature (if exists)
- rolling 7-day humidity mean
- month
- day of week

Target:
- next-day humidity (shift -1)

Ensures no future leakage. Returns X, y.
"""
import os
import pandas as pd

def build_features(gold_csv_path=None):
    """
    Reads gold.csv, builds features for next-day humidity prediction.
    Args:
        gold_csv_path (str): Path to gold.csv. If None, uses default location.
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector (next-day humidity)
    """
    if gold_csv_path is None:
        gold_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gold', 'gold.csv')
        gold_csv_path = os.path.abspath(gold_csv_path)
    df = pd.read_csv(gold_csv_path, parse_dates=True)
    # Try to auto-detect datetime column
    datetime_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.sort_values(datetime_col)
    # Target: next-day humidity
    if 'humidity' not in df.columns:
        raise ValueError('humidity column not found in gold.csv')
    df['target'] = df['humidity'].shift(-1)
    # Features
    df['lag1_humidity'] = df['humidity'].shift(1)
    if 'temperature' in df.columns:
        df['lag1_temperature'] = df['temperature'].shift(1)
    df['humidity_rolling7'] = df['humidity'].rolling(window=7, min_periods=1).mean().shift(1)
    if datetime_col:
        df['month'] = df[datetime_col].dt.month
        df['dayofweek'] = df[datetime_col].dt.dayofweek
    # Drop last row (target is NaN)
    df = df.iloc[:-1]
    # Drop rows with any NaN (from shifting/rolling)
    df = df.dropna()
    feature_cols = ['lag1_humidity', 'humidity_rolling7', 'month', 'dayofweek']
    if 'lag1_temperature' in df.columns:
        feature_cols.insert(1, 'lag1_temperature')
    X = df[feature_cols]
    y = df['target']
    return X, y
