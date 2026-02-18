import os
import pytest
from src import data_version

def test_get_data_version():
    # Setup: create a temporary gold.csv file
    gold_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'gold')
    os.makedirs(gold_dir, exist_ok=True)
    gold_path = os.path.join(gold_dir, 'gold.csv')
    with open(gold_path, 'w', encoding='utf-8') as f:
        f.write('col1,col2\n1,2\n3,4\n')

    data_version_id, row_count, dataset_path = data_version.get_data_version_id()
    print(f"data_version_id: {data_version_id}")
    print(f"row_count: {row_count}")
    print(f"dataset_path: {dataset_path}")
    assert data_version_id is not None and data_version_id != ""
    assert row_count == 2
    assert dataset_path.endswith('gold.csv')

    # Cleanup
    os.remove(gold_path)
