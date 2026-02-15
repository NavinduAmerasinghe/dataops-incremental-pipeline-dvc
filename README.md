# mlops-assignment3-dataops

Lightweight DataOps pipeline example (split → bronze → validate/silver → transform → gold).

Quick commands

- Install deps: `pip install -r requirements.txt`
- Run full pipeline (sequential):
  - `python src/split_into_batches.py`
  - `python src/ingest_bronze.py`
  - `python src/validate_silver.py`
  - `python src/transform_silver.py`
  - `python src/build_gold.py`
- Run tests: `pytest -q`

Structure

Below is the repository file structure (key files and folders):

```
├─ dvc.yaml
├─ params.yaml
├─ requirements.txt
├─ README.md
├─ dvcstore/
├─ evidence/
│  ├─ dvc.lock
│  ├─ dvc.yaml
│  ├─ gold_snapshot.csv
│  ├─ manifest.csv
│  ├─ dvc_status.txt
│  ├─ folder_tree.txt
│  ├─ git_log.txt
│  └─ README.md
├─ data/
│  ├─ _staging_batches/
│  │  ├─ batch_01.csv
│  │  ├─ batch_02.csv
│  │  ├─ batch_03.csv
│  │  ├─ batch_04.csv
│  │  └─ batch_05.csv
│  ├─ incoming/
│  │  ├─ batch_01.csv
│  │  ├─ batch_02.csv
│  │  ├─ batch_03.csv
│  │  ├─ batch_04.csv
│  │  └─ batch_05.csv
│  ├─ bronze/
│  │  ├─ raw_batch_01.csv
│  │  ├─ raw_batch_01.csv.dvc
│  │  ├─ raw_batch_02.csv
│  │  ├─ raw_batch_02.csv.dvc
│  │  ├─ raw_batch_03.csv
│  │  ├─ raw_batch_03.csv.dvc
│  │  ├─ raw_batch_04.csv
│  │  ├─ raw_batch_04.csv.dvc
│  │  ├─ raw_batch_05.csv
│  │  ├─ raw_batch_05.csv.dvc
│  │  ├─ bronze_all.csv
│  │  ├─ bronze_all.csv.dvc
│  │  └─ manifest.csv
│  ├─ silver/
│  │  ├─ rejected_rows.csv
│  │  ├─ silver_all.csv
│  │  └─ transformed_silver.csv
│  └─ gold/
│     ├─ gold.csv
│     ├─ gold.csv.dvc
│     ├─ train.csv
│     └─ test.csv
├─ src/
│  ├─ split_into_batches.py
│  ├─ ingest_bronze.py
│  ├─ validate_silver.py
│  ├─ transform_silver.py
│  ├─ build_gold.py
│  ├─ simulate_incremental_ingest.py
│  ├─ collect_evidence.py
│  ├─ save_evidence.py
│  └─ write_fallback_dvc.py
└─ tests/
   ├─ test_incremental_ingest.py
   ├─ test_quality.py
   └─ test_validate_checks.py
```

- `data/` contains pipeline inputs and outputs (incoming → bronze → silver → gold).
- `src/` holds the pipeline scripts used by the example DVC pipeline.
- `tests/` contains unit/quality checks run with `pytest`.
- `evidence/` stores assignment evidence and snapshots (DVC/status files).
- `dvcstore/` is the local DVC store used for examples.

Design notes

- `split_into_batches.py` creates 5 example batches (synthetic) in `data/incoming/`.
- `ingest_bronze.py` copies incoming batches into `data/bronze/` and appends a master CSV.
- `validate_silver.py` validates rows and writes cleaned data to `data/silver/`.
- `transform_silver.py` produces feature-engineered `transformed_silver.csv`.
- `build_gold.py` creates `data/gold/gold.csv` and `train/test` splits.

Use this as a starting scaffold for DataOps / MLOps exercises.
