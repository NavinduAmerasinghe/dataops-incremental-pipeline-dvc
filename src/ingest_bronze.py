"""
ingest_bronze.py
- Moves CSV batches from `data/incoming/` into `data/bronze/` (append-only store)
- Maintains `data/bronze/bronze_all.csv` (appended master file)
"""
from pathlib import Path
import shutil
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    incoming = repo_root / "data" / "incoming"
    bronze_dir = repo_root / "data" / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    master_path = bronze_dir / "bronze_all.csv"
    incoming_files = sorted(incoming.glob("*.csv"))
    if not incoming_files:
        print("No incoming batches found.")
        return

    for f in incoming_files:
        print(f"Ingesting {f.name} -> bronze/")
        # copy raw file into bronze/raw_<name>
        raw_copy = bronze_dir / f"raw_{f.name}"
        shutil.copy2(f, raw_copy)

        # append to master bronze file
        df = pd.read_csv(f)
        header = not master_path.exists()
        df.to_csv(master_path, mode="a", header=header, index=False)

        # remove incoming file (simulate move/consume)
        f.unlink()

    print(f"Updated master bronze file at: {master_path}")


if __name__ == "__main__":
    main()
