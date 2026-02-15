import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd


def run(script):
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / "src" / script)], check=True, cwd=repo)


def write_bronze(rows, repo):
    bronze_dir = repo / "data" / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    master = bronze_dir / "bronze_all.csv"
    pd.DataFrame(rows).to_csv(master, index=False)
    return master


def test_duplicate_timestamp_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    # prepare clean bronze
    bdir = repo / "data" / "bronze"
    if bdir.exists():
        shutil.rmtree(bdir)

    rows = [
        {"id": 1, "feature_num": 10.0, "feature_cat": "A", "target": 0, "timestamp": "2024-01-01 00:00:00"},
        {"id": 2, "feature_num": 12.0, "feature_cat": "B", "target": 1, "timestamp": "2024-01-01 00:01:00"},
        # duplicate timestamp
        {"id": 3, "feature_num": 13.0, "feature_cat": "A", "target": 0, "timestamp": "2024-01-01 00:01:00"},
    ]
    write_bronze(rows, repo)

    run("validate_silver.py")
    rejected = pd.read_csv(repo / "data" / "silver" / "rejected_rows.csv")
    assert 2 in set(rejected["id"].astype(int).tolist()) or 3 in set(rejected["id"].astype(int).tolist())


def test_date_continuity_gap_marked(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bdir = repo / "data" / "bronze"
    if bdir.exists():
        shutil.rmtree(bdir)

    rows = [
        {"id": 1, "feature_num": 10.0, "feature_cat": "A", "target": 0, "timestamp": "2024-01-01 00:00:00"},
        {"id": 2, "feature_num": 12.0, "feature_cat": "B", "target": 1, "timestamp": "2024-01-01 00:01:00"},
        # large gap
        {"id": 3, "feature_num": 13.0, "feature_cat": "A", "target": 0, "timestamp": "2024-01-01 01:10:00"},
    ]
    write_bronze(rows, repo)
    run("validate_silver.py")
    rejected = pd.read_csv(repo / "data" / "silver" / "rejected_rows.csv")
    # row with id 3 should be rejected due to large gap
    assert 3 in set(rejected["id"].astype(int).tolist())


def test_value_range_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bdir = repo / "data" / "bronze"
    if bdir.exists():
        shutil.rmtree(bdir)

    rows = [
        {"id": 1, "feature_num": 10.0, "feature_cat": "A", "target": 0, "timestamp": "2024-01-01 00:00:00"},
        {"id": 2, "feature_num": 1000.0, "feature_cat": "B", "target": 1, "timestamp": "2024-01-01 00:01:00"},
    ]
    write_bronze(rows, repo)
    run("validate_silver.py")
    rejected = pd.read_csv(repo / "data" / "silver" / "rejected_rows.csv")
    assert 2 in set(rejected["id"].astype(int).tolist())
