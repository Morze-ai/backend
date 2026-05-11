# Implementation status - seaData/backend

Short summary of what is implemented in the codebase and what still needs work to complete the pipeline described in docs/checklist.md.

## Implemented

- **Data ingestion and scripts**: basic dataset fetch and normalization scripts exist (examples: [scripts/fetch_imgw_dataset.py](scripts/fetch_imgw_dataset.py), [scripts/normalize_raw_datasets.py](scripts/normalize_raw_datasets.py)).
- **Preprocessing and cleaning**: cleaning utilities and missing-value handling in [scripts/clean_missing_values.py](scripts/clean_missing_values.py) and [src/data/preprocessing.py](src/data/preprocessing.py).
- **Synchronization and resampling**: synchronization utilities in [src/data/synchronization.py](src/data/synchronization.py) and resampling helpers used by the pipeline (resampled datasets in data/processed/).
- **Dataset abstraction**: dataset handling in [src/data/dataset.py](src/data/dataset.py).
- **Models and experiments**: implemented model definitions and experiments for linear, logistic, and MLP approaches: [src/models/linear.py](src/models/linear.py), [src/models/logistic_regression.py](src/models/logistic_regression.py), [src/models/mlp.py](src/models/mlp.py); experiment drivers in [src/experiments/](src/experiments/).
- **Training loop and trainer**: central training logic in [src/training/trainer.py](src/training/trainer.py) and CLI entry points for training in [src/cli/train_model.py](src/cli/train_model.py).
- **CLI**: command-line interface commands for common tasks in [src/cli/](src/cli/) (fetch, preprocess, train, predict, evaluate, visualize).
- **Visualization**: plotting helpers in [src/visualization/plots.py](src/visualization/plots.py) and CLI visualization support in [src/cli/visualize.py](src/cli/visualize.py).
- **Utilities**: configuration, IO, logging, seeding, and runtime helpers in [src/utils/](src/utils/).
- **Tests**: unit tests covering config, experiments, preprocessing, trainer, domain feature engineering, and lag feature engineering in [tests/](tests/).
- **Domain feature engineering**: implemented in [src/data/feature_engineering.py](src/data/feature_engineering.py), including rain sums, temperature delta, thaw flag, soil saturation proxy, and wind component features.
- **Lag features**: implemented in [src/data/feature_engineering.py](src/data/feature_engineering.py) with automatic integration in [src/experiments/base.py](src/experiments/base.py), including dynamic lag generation, warmup row dropping, and automatic lag column registration.
- **Seasonal features**: implemented in [src/data/feature_engineering.py](src/data/feature_engineering.py) (month, day-of-year, day-of-week, hour-of-day, season, growing-season flag).
- **Rolling aggregates**: implemented in [src/data/feature_engineering.py](src/data/feature_engineering.py) (mean, max, min, std over configurable windows).

## Partially implemented / needs verification

- **Aggregations for non-water-level datasets**: water-level aggregation exists; meteorological aggregation pipelines are scaffolding but need dataset wiring and verification.
- **Experiment registry and reproducibility**: registry exists ([src/experiments/registry.py](src/experiments/registry.py)), but experiment comparison and reporting need enrichment.
- **Feature engineering configurability**: lag toggles and lag length are config-driven, but rolling and seasonal toggles are not yet fully wired into end-to-end preprocessing flow.

## Not implemented / TODO (high priority)

- **Event detection rules (O1-O4)**: implement the rule-based sensor system that labels episodes and produces readable messages.
- **Confidence estimation and interpretability**: probability calibration, SHAP analysis, and confidence score per event.
- **Model evaluation and validation**: temporal split (2021-2023 train, 2024-2025 test), recall/precision optimization, onset error metric implementation.
- **Reporting outputs**: automated CSV and PDF generators summarizing events, factors, and metrics.
- **Notebook pipeline and user guide**: consolidate the pipeline into a reproducible notebook and example walkthrough.

## Recommended next steps

- Wire feature engineering toggles from config into the full preprocessing flow (rolling and seasonal toggles alongside lag toggles).
- Add an events module to encode O1-O4 rules and a small evaluation harness to compare rule-based vs model predictions.
- Add SHAP analysis and strengthen the [src/cli/explain.py](src/cli/explain.py) workflow to produce feature-impact reports.
- Add a short docs/USAGE.md showing how to run common commands and how to run tests.

## Quick commands

- Run tests: pytest -q
- Run CLI help: python -m src.cli.fetch_data --help (or use the entry points in src/cli/)
