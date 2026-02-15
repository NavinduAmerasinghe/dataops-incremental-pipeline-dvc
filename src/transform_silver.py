"""
transform_silver.py
- Reads `data/silver/silver_all.csv`, applies lightweight feature engineering and normalization,
  and writes `data/silver/transformed_silver.csv` (ML-ready features).
"""
from pathlib import Path
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    silver_in = repo_root / "data" / "silver" / "silver_all.csv"
    silver_out = repo_root / "data" / "silver" / "transformed_silver.csv"

    if not silver_in.exists():
        print("No validated silver dataset found. Run validate_silver.py first.")
        return

    df = pd.read_csv(silver_in)

    # parse timestamp and extract simple temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(-1).astype(int)

    # categorical encoding (simple)
    df["feature_cat_code"] = pd.Categorical(df["feature_cat"]).codes

    # numeric scaling (min-max)
    mn = df["feature_num"].min()
    mx = df["feature_num"].max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        df["feature_num_scaled"] = 0.0
    else:
        df["feature_num_scaled"] = ((df["feature_num"] - mn) / (mx - mn)).clip(0, 1)

    df.to_csv(silver_out, index=False)
    print(f"Wrote transformed silver -> {silver_out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
