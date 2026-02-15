"""
build_gold.py
- Produces `data/gold/gold.csv` (ML-ready dataset) and train/test splits.
"""
from pathlib import Path
import pandas as pd
import yaml


def load_params(repo_root: Path):
    p = repo_root / "params.yaml"
    if not p.exists():
        return {"test_size": 0.2, "random_seed": 42}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {"test_size": float(data.get("test_size", 0.2)), "random_seed": int(data.get("random_seed", 42))}


def main():
    repo_root = Path(__file__).resolve().parents[1]
    transformed = repo_root / "data" / "silver" / "transformed_silver.csv"
    gold_dir = repo_root / "data" / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    if not transformed.exists():
        print("No transformed silver found. Run transform_silver.py first.")
        return

    df = pd.read_csv(transformed)
    # select a subset of ML-ready columns
    cols = [c for c in ["id", "feature_num_scaled", "feature_cat_code", "day_of_week", "target"] if c in df.columns]
    gold = df[cols].copy()
    gold_path = gold_dir / "gold.csv"
    gold.to_csv(gold_path, index=False)

    params = load_params(repo_root)
    test_size = params["test_size"]
    seed = params["random_seed"]

    # simple random split
    test = gold.sample(frac=test_size, random_state=seed)
    train = gold.drop(test.index)
    train.to_csv(gold_dir / "train.csv", index=False)
    test.to_csv(gold_dir / "test.csv", index=False)

    print(f"Built gold dataset: {gold_path} (rows={len(gold)})")
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")


if __name__ == "__main__":
    main()
