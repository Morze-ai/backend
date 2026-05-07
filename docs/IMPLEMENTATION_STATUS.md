# Implementation status — seaData/backend

Short summary of what's implemented in the codebase and what still needs work to complete the pipeline described in docs/checklist.md.

## Implemented

- **Data ingestion & scripts**: basic dataset fetch and normalization scripts exist (examples: [scripts/fetch_imgw_dataset.py](scripts/fetch_imgw_dataset.py), [scripts/normalize_raw_datasets.py](scripts/normalize_raw_datasets.py)).
- **Preprocessing & cleaning**: cleaning utilities and missing-value handling in [scripts/clean_missing_values.py](scripts/clean_missing_values.py) and [src/data/preprocessing.py](src/data/preprocessing.py).
- **Synchronization & resampling**: synchronization utilities in [src/data/synchronization.py](src/data/synchronization.py) and resampling helpers used by the pipeline (resampled datasets in `data/processed/`).
- **Dataset abstraction**: dataset handling in [src/data/dataset.py](src/data/dataset.py).
- **Models & experiments**: implemented model definitions and experiments for linear, logistic and MLP approaches: [src/models/linear.py](src/models/linear.py), [src/models/logistic_regression.py](src/models/logistic_regression.py), [src/models/mlp.py](src/models/mlp.py); experiment drivers in [src/experiments/](src/experiments/).
- **Training loop & trainer**: central training logic in [src/training/trainer.py](src/training/trainer.py) and CLI entrypoints for training in [src/cli/train_model.py](src/cli/train_model.py).
- **CLI**: command-line interface commands for common tasks in [src/cli/](src/cli/) (fetch, preprocess, train, predict, evaluate, visualize).
- **Visualization**: plotting helpers in [src/visualization/plots.py](src/visualization/plots.py) and CLI visualization support in [src/cli/visualize.py](src/cli/visualize.py).
- **Utilities**: configuration, IO, logging, seeding and runtime helpers in [src/utils/](src/utils/).
- **Tests**: unit tests covering config, experiments, preprocessing and trainer in [tests/](tests/).

## Partially implemented / needs verification

- **Aggregations for non-water-level datasets**: water-level aggregation exists; meteorological aggregation pipelines are scaffolding but need dataset wiring and verification.
- **Experiment registry & reproducibility**: registry exists ([src/experiments/registry.py](src/experiments/registry.py)), but experiment comparison/reporting needs enrichment.

## Not implemented / TODO (high priority)

- **Feature engineering**: implement engineered features described in checklist (rain_xh_sum, temp_delta_24h, thaw_flag, soil-saturation indicators).
- **Lag features**: generate lag features (rain_lag_1h..72h, temp_lag, pressure_lag) and integrate into dataset pipeline.
- **Seasonal features**: month, day-of-week, seasonal encodings and calendar features.
- **Event detection rules (O1–O4)**: implement the rule-based sensor system that labels episodes and produces readable messages.
- **Confidence estimation & interpretability**: probability calibration, SHAP analysis, and confidence score per event.
- **Model evaluation & validation**: temporal split (2021–2023 train, 2024–2025 test), recall/precision optimization, onset error metric implementation.
- **Reporting outputs**: automated CSV and PDF generators summarizing events, factors and metrics.
- **Notebook pipeline & user guide**: consolidate the pipeline into a reproducible notebook and example walkthrough.

## Recommended next steps

- Implement feature engineering and lags in `src/data/preprocessing.py` and/or a new `src/data/feature_engineering.py` module.
- Add an `events/` module to encode O1–O4 rules and a small evaluation harness to compare rule-based vs model predictions.
- Add SHAP analysis and an `explain/` CLI command to produce feature-impact reports.
- Add a short `docs/USAGE.md` showing how to run common commands and how to run tests: `pytest -q`.

## Quick commands

- Run tests: `pytest -q`
- Run CLI help: `python -m src.cli.fetch_data --help` (or use the entrypoints in `src/cli/`)
