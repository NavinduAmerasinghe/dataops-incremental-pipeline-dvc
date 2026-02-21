
import os
import hashlib
import json
from typing import Optional, Dict

def get_data_version_id(dvc_lock_path: str = "dvc.lock", dataset_path: str = "data/gold/gold.csv") -> Dict[str, str]:
    """
    Returns a dictionary with data_version_id, row_count, and dataset_path.
    If dvc.lock exists, extract a stable SHA/MD5 identifier from it.
    Otherwise, hash the content of the dataset file.
    """
    data_version_id = None
    row_count = None
    if os.path.exists(dvc_lock_path):
        try:
            with open(dvc_lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            # Try to extract hash from dvc.lock (supports both md5 and sha256)
            for stage in lock_data.get("stages", {}).values():
                outs = stage.get("outs", [])
                for out in outs:
                    if "md5" in out:
                        data_version_id = out["md5"]
                    elif "sha256" in out:
                        data_version_id = out["sha256"]
                    if data_version_id:
                        break
                if data_version_id:
                    break
        except Exception as e:
            print(f"[WARN] Failed to load dvc.lock as JSON: {e}. Falling back to dataset hash.")
    if not data_version_id:
        # Fallback: hash the dataset file
        if os.path.exists(dataset_path):
            hasher = hashlib.md5()
            with open(dataset_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            data_version_id = hasher.hexdigest()
        else:
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    # Get row count
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # Exclude header
    else:
        row_count = None
    return {
        "data_version_id": data_version_id,
        "row_count": row_count,
        "dataset_path": dataset_path
    }

def get_git_commit_hash() -> Optional[str]:
    """
    Returns the current git commit hash if git is available, else None.
    """
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None

# Example unit test (can be placed in tests/test_data_version.py)
def _example_unit_test():
    """
    Example unit test for get_data_version_id().
    """
    result = get_data_version_id()
    assert "data_version_id" in result
    assert "row_count" in result
    assert "dataset_path" in result
    print("Test passed.")

if __name__ == "__main__":
    print(get_data_version_id())
    print("Git commit hash:", get_git_commit_hash())
