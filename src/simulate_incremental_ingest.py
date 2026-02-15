"""
simulate_incremental_ingest.py
- Split `data/gold/train.csv` into N time-ordered batches and simulate incremental
  arrival: place one batch at a time into `data/incoming/` and run the pipeline stages
  (ingest -> validate -> transform -> build) for each arrival.

Usage: python src/simulate_incremental_ingest.py --n 5
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import pandas as pd
import numpy as np
import yaml
from datetime import datetime


def split_into_batches(df: pd.DataFrame, n: int):
    # preserve order (time-order assumed in 'timestamp' or 'id' if present)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    elif "id" in df.columns:
        df = df.sort_values("id")

    # compute chunk sizes and slice with iloc to preserve DataFrame dtype/index
    total = len(df)
    if total == 0:
        return [df.copy() for _ in range(n)]
    base = total // n
    rem = total % n
    parts = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        end = start + size
        parts.append(df.iloc[start:end].copy())
        start = end
    return parts


def run_script(repo_root: Path, script: str):
    cmd = [sys.executable, str(repo_root / "src" / script)]
    subprocess.run(cmd, check=True, cwd=repo_root)


def main(n_batches: int, source: Path, staging_dir: Path, reset: bool):
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / source
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)
    parts = split_into_batches(df, n_batches)

    # prepare staging area for batches
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for i, part in enumerate(parts, start=1):
        out = staging_dir / f"batch_{i:02d}.csv"
        part.to_csv(out, index=False)
        print(f"Prepared {out} ({len(part)} rows)")

    # optionally reset generated data folders so simulation is deterministic
    data_root = repo_root / "data"
    bronze_dir = data_root / "bronze"
    if reset:
        for d in ["incoming", "bronze", "silver", "gold"]:
            p = data_root / d
            if p.exists():
                shutil.rmtree(p)

    # ensure incoming exists
    (data_root / "incoming").mkdir(parents=True, exist_ok=True)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = bronze_dir / "manifest.csv"
    # Manifest columns: batch_name,rows_in_batch,raw_path,dvc_checksum,git_commit,recorded_at
    if not manifest_path.exists():
        with manifest_path.open("w", encoding="utf-8") as m:
            m.write("batch_name,rows_in_batch,raw_path,dvc_checksum,git_commit,recorded_at\n")

    # process each staged batch in order
    staged = sorted(staging_dir.glob("batch_*.csv"))
    for batch in staged:
        incoming_dest = data_root / "incoming" / batch.name
        shutil.copy2(batch, incoming_dest)
        print(f"Arrived batch: {batch.name} -> incoming/")

        # run ingest to move incoming -> bronze and update master
        run_script(repo_root, "ingest_bronze.py")

        # identify raw copy created by ingest and record row count
        raw_copy = bronze_dir / f"raw_{batch.name}"
        checksum = ""
        rows_in_batch = 0
        try:
            import pandas as _pd
            rows_in_batch = len(_pd.read_csv(raw_copy))
        except Exception:
            # fallback to the staged file size
            try:
                import csv
                with (data_root / 'incoming' / batch.name).open('r', encoding='utf-8') as f:
                    rows_in_batch = sum(1 for _ in f) - 1
            except Exception:
                rows_in_batch = 0

        # try to add to DVC (creates <file>.dvc) when dvc is available
        try:
            subprocess.run([sys.executable, "-m", "dvc", "add", str(raw_copy)], check=True, cwd=repo_root)
        except Exception:
            print("Warning: dvc add failed or not available; will write fallback .dvc with md5 checksum")

        # read checksum from .dvc if produced, otherwise compute md5 and write minimal .dvc
        dvc_file = Path(str(raw_copy) + ".dvc")
        if dvc_file.exists():
            try:
                with dvc_file.open("r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                outs = doc.get("outs") or []
                if outs and isinstance(outs, list):
                    md = outs[0].get("md5") or outs[0].get("etag") or outs[0].get("checksum")
                    checksum = md or ""
            except Exception:
                checksum = ""
        else:
            try:
                import hashlib

                h = hashlib.md5()
                with raw_copy.open("rb") as rf:
                    for chunk in iter(lambda: rf.read(8192), b""):
                        h.update(chunk)
                checksum = h.hexdigest()
                dvc_doc = {"outs": [{"path": raw_copy.name, "md5": checksum}]}
                try:
                    with dvc_file.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(dvc_doc, f)
                except Exception:
                    pass
            except Exception:
                checksum = ""

        # get git commit hash if available
        commit = ""
        try:
            p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root)
            if p.returncode == 0:
                commit = p.stdout.strip()
        except Exception:
            commit = ""

        # append to manifest (persistent provenance record)
        with manifest_path.open("a", encoding="utf-8") as m:
            m.write(f"{batch.name},{rows_in_batch},{raw_copy.as_posix()},{checksum},{commit},{datetime.utcnow().isoformat()}\n")

        # subsequent pipeline stages for this arrival
        run_script(repo_root, "validate_silver.py")
        run_script(repo_root, "transform_silver.py")
        run_script(repo_root, "build_gold.py")

        # If DVC is available, track layer outputs and commit per-batch
        try:
            # add updated layers to DVC
            subprocess.run([sys.executable, "-m", "dvc", "add", str(bronze_dir)], check=True, cwd=repo_root)
            subprocess.run([sys.executable, "-m", "dvc", "add", str(data_root / "silver")], check=True, cwd=repo_root)
            subprocess.run([sys.executable, "-m", "dvc", "add", str(data_root / "gold")], check=True, cwd=repo_root)

            # git-add the produced .dvc files and dvc.lock/yaml if present, then commit
            to_add = []
            for p in [bronze_dir, data_root / "silver", data_root / "gold"]:
                dvc_file = Path(str(p) + ".dvc")
                if dvc_file.exists():
                    to_add.append(str(dvc_file.relative_to(repo_root)))
            for fname in ["dvc.lock", "dvc.yaml", ".gitignore"]:
                f = repo_root / fname
                if f.exists():
                    to_add.append(str(f.relative_to(repo_root)))

            if to_add:
                subprocess.run(["git", "add"] + to_add, check=True, cwd=repo_root)
                msg = f"Batch {batch.stem}: update Bronze/Silver/Gold"
                subprocess.run(["git", "commit", "-m", msg], check=True, cwd=repo_root)
        except Exception:
            # DVC/git not available or commit failed; continue without failing the pipeline
            pass

        # report counts after this batch
        bronze_master = data_root / "bronze" / "bronze_all.csv"
        silver_master = data_root / "silver" / "silver_all.csv"
        gold_master = data_root / "gold" / "gold.csv"
        print("State after processing batch:")
        if bronze_master.exists():
            print("  bronze_all rows:", len(pd.read_csv(bronze_master)))
        if silver_master.exists():
            print("  silver_all rows:", len(pd.read_csv(silver_master)))
        if gold_master.exists():
            print("  gold rows:", len(pd.read_csv(gold_master)))

    print("\nSimulation complete. Processed all batches incrementally.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of batches to split into")
    parser.add_argument("--source", type=str, default="data/gold/train.csv", help="Source CSV to split")
    parser.add_argument("--staging", type=str, default="data/_staging_batches", help="Where to write batch files before arrival")
    parser.add_argument("--no-reset", dest="reset", action="store_false", help="Do not clear generated data dirs before simulation")
    parser.set_defaults(reset=True)
    args = parser.parse_args()

    main(args.n, Path(args.source), Path(args.staging), args.reset)