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

- `data/` — incoming / bronze / silver / gold
- `src/` — pipeline scripts
- `tests/` — pytest quality checks
- `dvc.yaml` & `params.yaml` — example pipeline and parameters

Design notes

- `split_into_batches.py` creates 5 example batches (synthetic) in `data/incoming/`.
- `ingest_bronze.py` copies incoming batches into `data/bronze/` and appends a master CSV.
- `validate_silver.py` validates rows and writes cleaned data to `data/silver/`.
- `transform_silver.py` produces feature-engineered `transformed_silver.csv`.
- `build_gold.py` creates `data/gold/gold.csv` and `train/test` splits.

Use this as a starting scaffold for DataOps / MLOps exercises.
