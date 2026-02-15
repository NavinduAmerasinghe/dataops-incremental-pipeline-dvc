# mlops-assignment3-dataops

## Project Overview

This project demonstrates a lightweight DataOps pipeline for incremental data ingestion, validation, transformation, and preparation for machine learning workflows. The pipeline follows a typical data engineering lifecycle: splitting raw data into batches, ingesting into a bronze layer, validating and cleaning into a silver layer, transforming features, and finally producing a gold dataset ready for ML modeling. The project is designed for educational purposes, focusing on reproducibility, modularity, and automation using DVC (Data Version Control) and Python scripts.

## Repository Contents

- **Source code:** `src/`
- **Configuration files:** `dvc.yaml`, `params.yaml`, `requirements.txt`
- **Pipeline definitions:** `dvc.yaml` and scripts in `src/`
- **Test scripts:** `tests/`
- **README (with execution flow and assumptions):** `README.md`

## What's Covered
- Incremental data ingestion and batch processing
- Data validation and cleaning (quality checks, row rejection)
- Feature engineering and transformation
- Train/test split for ML tasks
- Reproducible pipelines with DVC
- Evidence collection for pipeline runs

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
3. **(Optional) Install DVC:**
  ```bash
  pip install dvc
  ```

## How to Run the Pipeline

You can run the pipeline step-by-step or use DVC to orchestrate the workflow.

**Manual (sequential) execution:**
```bash
python src/split_into_batches.py
python src/ingest_bronze.py
python src/validate_silver.py
python src/transform_silver.py
python src/build_gold.py
```

**Run all tests:**
```bash
pytest -q
```

**With DVC:**
```bash
dvc repro
```

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

## Execution Flow and Assumptions

**Execution Flow:**
1. The pipeline starts by splitting raw data into five synthetic batches (`split_into_batches.py`).
2. Each batch is incrementally ingested into the bronze layer (`ingest_bronze.py`), where all raw batches are appended to a master CSV.
3. Data is validated and cleaned in the silver layer (`validate_silver.py`), with invalid rows rejected and logged.
4. Feature engineering and transformation are performed on the cleaned data (`transform_silver.py`).
5. The final gold dataset is built and split into train/test sets for ML tasks (`build_gold.py`).

**Assumptions:**
- The pipeline expects CSV files with a consistent schema in the `data/incoming/` directory.
- All scripts are run in order, and each step depends on the successful completion of the previous one.
- Python 3.8+ and all dependencies in `requirements.txt` are installed.
- DVC is used for reproducibility, but scripts can be run manually if DVC is not available.
- The data is synthetic and for demonstration; real-world data may require schema adjustments and more robust validation.

## Expected Output
- **Gold dataset:** `data/gold/gold.csv` — fully cleaned, feature-engineered, and ready for ML modeling.
- **Train/test splits:** `data/gold/train.csv`, `data/gold/test.csv`
- **Evidence:** Run logs, DVC status, and snapshots in `evidence/`
- **Intermediate artifacts:** All batch, bronze, and silver files for traceability

## Conclusion

This project provides a hands-on scaffold for learning and practicing DataOps and MLOps principles. By following the pipeline, you will understand how to structure data workflows, enforce data quality, and ensure reproducibility using DVC. The modular scripts and clear data lineage make it easy to extend or adapt for more complex real-world scenarios.

