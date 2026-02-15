"""
split_into_batches.py
- Generates a synthetic dataset and writes it to `data/incoming/` as N CSV batches.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import yaml


def load_params(params_path: Path):
    defaults = {"n_batches": 5, "rows_per_batch": 100, "random_seed": 42}
    if not params_path.exists():
        return defaults
    with params_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    defaults.update({k: data.get(k, v) for k, v in defaults.items()})
    return defaults


def make_synthetic(total_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": np.arange(1, total_rows + 1),
            "feature_num": rng.normal(loc=50, scale=15, size=total_rows).round(3),
            "feature_cat": rng.choice(["A", "B", "C"], size=total_rows, p=[0.5, 0.3, 0.2]),
            "target": rng.integers(0, 2, size=total_rows),
            "timestamp": pd.date_range(start="2024-01-01", periods=total_rows, freq="min").astype(str),
        }
    )
    return df


def main(base_path: Path = None):
    repo_root = Path(__file__).resolve().parents[1] if base_path is None else Path(base_path)
    params = load_params(repo_root / "params.yaml")

    incoming = repo_root / "data" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)

    n_batches = int(params["n_batches"])
    rows_per_batch = int(params["rows_per_batch"])
    seed = int(params["random_seed"])

    total = n_batches * rows_per_batch
    df = make_synthetic(total, seed)

    for i in range(n_batches):
        start = i * rows_per_batch
        end = start + rows_per_batch
        batch = df.iloc[start:end].copy()
        out_path = incoming / f"batch_{i+1:02d}.csv"
        batch.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(batch)} rows)")

    print(f"Created {n_batches} batches in '{incoming}' (total rows={total})")


if __name__ == "__main__":
    main()
