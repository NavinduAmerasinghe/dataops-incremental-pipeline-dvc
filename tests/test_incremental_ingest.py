import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd


def test_incremental_ingest_from_train(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    # Recreate required directories after cleanup
    incoming_dir = repo / "data" / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    # Clean up all relevant data folders and files
    for d in ["incoming", "bronze", "silver", "gold", "_staging_batches"]:
        p = repo / "data" / d
        if p.exists():
            shutil.rmtree(p)

    # Ensure params.yaml has enough rows for 5 batches
    params_path = repo / "params.yaml"
    import yaml
    params = {"n_batches": 5, "rows_per_batch": 100, "random_seed": 42}
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f)

    # create a fresh bronze master to split from
    subprocess.run([sys.executable, str(repo / "src" / "split_into_batches.py")], check=True, cwd=repo)
    subprocess.run([sys.executable, str(repo / "src" / "ingest_bronze.py")], check=True, cwd=repo)

    source = repo / "data" / "bronze" / "bronze_all.csv"
    assert source.exists(), "Source bronze_all.csv must exist for this test"
    src_df = pd.read_csv(source)
    assert len(src_df) == 500, "Synthetic data should have 500 rows (5 batches × 100)"

    # run the incremental ingestion simulation (split the bronze master into 5 batches)
    subprocess.run(
        [
            sys.executable,
            str(repo / "src" / "simulate_incremental_ingest.py"),
            "--source",
            "data/bronze/bronze_all.csv",
            "--n",
            "5",
        ],
        check=True,
        cwd=repo,
    )

    staging = repo / "data" / "_staging_batches"
    assert staging.exists()
    batches = sorted(staging.glob("batch_*.csv"))
    assert len(batches) == 5, "Should create 5 batches"

    bronze_dir = repo / "data" / "bronze"
    assert bronze_dir.exists()
    raw_batches = list(bronze_dir.glob("raw_batch_*.csv"))
    assert len(raw_batches) == 5, "All 5 batches should have been ingested into bronze as raw files"

    bronze_master = bronze_dir / "bronze_all.csv"
    assert bronze_master.exists()
    assert len(pd.read_csv(bronze_master)) == len(src_df)

    gold_master = repo / "data" / "gold" / "gold.csv"
    assert gold_master.exists()
    # Gold should have at least as many rows as valid input (allowing for some validation rejections)
    gold_rows = len(pd.read_csv(gold_master))
    assert gold_rows > 0 and gold_rows <= len(src_df)

    # manifest must exist and contain one entry per batch
    manifest = bronze_dir / "manifest.csv"
    assert manifest.exists(), "Manifest must exist for DVC-tracked batches"
    mf = pd.read_csv(manifest)
    assert len(mf) >= 5, "Manifest should have at least one entry per ingested batch"
