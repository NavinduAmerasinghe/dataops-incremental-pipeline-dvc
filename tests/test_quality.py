import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd


def run(script_name: str):
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo / "src" / script_name)]
    subprocess.run(cmd, check=True, cwd=repo)


def test_end_to_end_pipeline(tmp_path):
    repo = Path(__file__).resolve().parents[1]

    # ensure a clean data directory
    data_dir = repo / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # run pipeline stages
    run("split_into_batches.py")
    run("ingest_bronze.py")
    run("validate_silver.py")
    run("transform_silver.py")
    run("build_gold.py")

    # assert outputs exist
    assert (repo / "data" / "bronze" / "bronze_all.csv").exists()
    assert (repo / "data" / "silver" / "transformed_silver.csv").exists()
    assert (repo / "data" / "gold" / "gold.csv").exists()
    assert (repo / "data" / "gold" / "train.csv").exists()
    assert (repo / "data" / "gold" / "test.csv").exists()

    # quality checks on gold
    gold = pd.read_csv(repo / "data" / "gold" / "gold.csv")
    assert not gold.isnull().any().any()
    assert set(gold["target"].unique()).issubset({0, 1})
    if "feature_num_scaled" in gold.columns:
        assert gold["feature_num_scaled"].min() >= 0 - 1e-8
        assert gold["feature_num_scaled"].max() <= 1 + 1e-8

    # train/test split sizes add up
    train = pd.read_csv(repo / "data" / "gold" / "train.csv")
    test = pd.read_csv(repo / "data" / "gold" / "test.csv")
    assert len(train) + len(test) == len(gold)


def test_validation_rejects_bad_row(tmp_path):
    """Ensure an obviously-bad row (invalid category/target/timestamp) is rejected."""
    repo = Path(__file__).resolve().parents[1]

    # start clean
    data_dir = repo / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # create normal data and ingest
    run("split_into_batches.py")
    run("ingest_bronze.py")

    # append a bad row to bronze_all.csv
    bronze = repo / "data" / "bronze" / "bronze_all.csv"
    with bronze.open("a", encoding="utf-8") as f:
        f.write("999,100.0,Z,2,not-a-timestamp\n")

    # validate and assert the bad row is rejected
    run("validate_silver.py")
    rejected = pd.read_csv(repo / "data" / "silver" / "rejected_rows.csv")
    assert 999 in set(rejected["id"].astype(int).tolist())
