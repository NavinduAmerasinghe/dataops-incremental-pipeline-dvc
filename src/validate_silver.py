"""
validate_silver.py
- Validates raw bronze data and writes cleaned rows to `data/silver/silver_all.csv`.
- Rejected rows (failing checks) are written to `data/silver/rejected_rows.csv`.
"""
from pathlib import Path
import pandas as pd
import yaml


def load_params(repo_root: Path):
    p = repo_root / "params.yaml"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_df(df: pd.DataFrame, required_cols):
    """Row-wise validation returning a boolean mask.

    - Keep the existing lightweight checks (nulls, types, uniqueness).
    - If `pandera` is available, run a schema validation and mark any
      rows that fail pandera checks as invalid as well.
    """
    df = df.copy()
    results = pd.Series(True, index=df.index)

    # required columns present
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        results &= df[c].notnull()

    # numeric parse check
    df["_feature_num"] = pd.to_numeric(df["feature_num"], errors="coerce")
    results &= df["_feature_num"].notnull()

    # target must be 0 or 1
    results &= df["target"].isin([0, 1])

    # timestamp parseable
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    results &= ts.notnull()

    # id uniqueness within the batch
    duplicated = df["id"].duplicated(keep=False)
    results &= ~duplicated

    # duplicate timestamps (mark all duplicates invalid)
    try:
        dup_ts = df["timestamp"].duplicated(keep=False)
        results &= ~dup_ts
    except Exception:
        pass

    # continuity check: flag rows following large gaps in timestamp
    try:
        # operate on parsed timestamps
        ts_sorted = ts.sort_values()
        diffs = ts_sorted.diff().dropna()
        if not diffs.empty:
            # choose the smallest positive diff as expected frequency (robust for small samples)
            positive = diffs[diffs > pd.Timedelta(0)]
            if not positive.empty:
                expected = positive.min()
            else:
                expected = diffs.median()
            # mark rows where the gap to previous is unusually large (>3x expected)
            large_gaps = diffs[diffs > 3 * expected]
            if not large_gaps.empty:
                gap_idx = large_gaps.index
                results.loc[gap_idx] = False
    except Exception:
        pass

    # value range checks for numeric features (configurable via params)
    try:
        # read bounds from params where available
        # default realistic range for feature_num
        min_val, max_val = -100.0, 200.0
        # if params provided via outer scope they should be passed; here we leave defaults
        out_of_range = (df["_feature_num"] < min_val) | (df["_feature_num"] > max_val)
        results &= ~out_of_range
    except Exception:
        pass

    # --- optional: pandera schema validation to catch subtle issues ---
    try:
        import pandera as pa
        from pandera.errors import SchemaErrors

        schema = pa.DataFrameSchema(
            {
                "id": pa.Column(pa.Int, checks=pa.Check(lambda s: s > 0)),
                "feature_num": pa.Column(pa.Float),
                "feature_cat": pa.Column(pa.String, checks=pa.Check.isin(["A", "B", "C"])),
                "target": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),
                "timestamp": pa.Column(pa.String, checks=pa.Check(lambda s: pd.to_datetime(s, errors="coerce").notnull())),
            },
            coerce=False,
        )
        try:
            schema.validate(df, lazy=True)
        except SchemaErrors as err:
            # mark pandera-failing rows as invalid
            failed_idx = err.failure_cases["index"].unique()
            results.loc[failed_idx] = False
    except Exception:
        # pandera not installed or unexpected error — ignore and keep core checks
        pass

    return results


def main():
    repo_root = Path(__file__).resolve().parents[1]
    bronze_master = repo_root / "data" / "bronze" / "bronze_all.csv"
    silver_dir = repo_root / "data" / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    if not bronze_master.exists():
        print("No bronze master file found — nothing to validate.")
        return

    df = pd.read_csv(bronze_master)
    params = load_params(repo_root)
    required = params.get("required_columns", ["id", "feature_num", "feature_cat", "target", "timestamp"])

    mask = validate_df(df, required)
    valid = df[mask].copy()
    invalid = df[~mask].copy()

    valid_path = silver_dir / "silver_all.csv"
    invalid_path = silver_dir / "rejected_rows.csv"

    valid.to_csv(valid_path, index=False)
    invalid.to_csv(invalid_path, index=False)

    print(f"Validation complete — valid: {len(valid)} rows, rejected: {len(invalid)} rows")


if __name__ == "__main__":
    main()
