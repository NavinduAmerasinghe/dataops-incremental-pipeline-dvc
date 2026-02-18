# Data versioning utility for Assignment 4 ModelOps

import os
import hashlib
import json
import subprocess


def get_data_version_id():
    """
    Returns a tuple: (data_version_id, row_count, dataset_path)
    """
    dvc_lock_path = os.path.join(os.path.dirname(__file__), '..', 'dvc.lock')
    gold_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gold', 'gold.csv')
    dvc_lock_path = os.path.abspath(dvc_lock_path)
    gold_path = os.path.abspath(gold_path)
    data_version_id = None
    row_count = None
    dataset_path = gold_path

    if os.path.exists(dvc_lock_path):
        with open(dvc_lock_path, 'r', encoding='utf-8') as f:
            try:
                dvc_lock = json.load(f)
                # Try to find gold.csv in dvc.lock
                for stage in dvc_lock.get('stages', {}).values():
                    outs = stage.get('outs', [])
                    for out in outs:
                        if 'gold/gold.csv' in out.get('path', ''):
                            data_version_id = out.get('md5') or out.get('hash')
                            break
            except Exception:
                pass
    if not data_version_id:
        # Fallback: hash the file content
        if os.path.exists(gold_path):
            hasher = hashlib.md5()
            with open(gold_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            data_version_id = hasher.hexdigest()
    # Row count
    if os.path.exists(gold_path):
        with open(gold_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1  # Exclude header
    return data_version_id, row_count, dataset_path

def get_git_commit_hash():
    """Returns the current git commit hash if available, else None."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None

# Example unit test (pytest style)
def test_get_data_version_id(tmp_path, monkeypatch):
    # Create a fake gold.csv
    gold = tmp_path / "gold.csv"
    gold.write_text("col1,col2\n1,2\n3,4\n")
    # Patch gold_path
    monkeypatch.setattr(__name__ + '.get_data_version_id', lambda: ("dummyhash", 2, str(gold)))
    data_version_id, row_count, dataset_path = get_data_version_id()
    assert data_version_id == "dummyhash"
    assert row_count == 2
    assert dataset_path.endswith("gold.csv")
