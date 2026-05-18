# Implementation Status - seaData/backend

Comprehensive audit of the project pipeline against [project_sheet.md](project_sheet.md) and [checklist.md](checklist.md) deliverables. Status reflects evidence from source code, tests, and CLI interfaces as of May 2026.

---

## Status Summary

| Phase | Status | Evidence |
| ------- | -------- | ---------- |
| **Data Ingestion** | ✅ Complete | [scripts/fetch_imgw_dataset.py](../scripts/fetch_imgw_dataset.py), [data/fetch_era5_data.py](../data/fetch_era5_data.py) |
| **Data Cleaning** | ✅ Complete | [src/data/preprocessing.py](../src/data/preprocessing.py) w/ missing-value strategies, normalization, and renaming |
| **Synchronization & Resampling** | ✅ Complete | [src/data/synchronization.py](../src/data/synchronization.py) w/ merge, alignment validation, daily aggregations |
| **Domain Features (O1-O4 inputs)** | ✅ Complete | [src/data/feature_engineering.py](../src/data/feature_engineering.py): rain sums, temp delta, thaw flag, soil saturation, wind components; ERA5 wind/pressure/SST [src/data/era5_processor.py](../src/data/era5_processor.py) |
| **ERA5 Data Integration** | ✅ Complete | [src/data/netcdf_loader.py](../src/data/netcdf_loader.py) + [src/data/era5_processor.py](../src/data/era5_processor.py): wind (u10→wind_u/v, direction), pressure (msl→hPa), SST (K→C) with hourly resampling, [scripts/prepare_training_data.py](../scripts/prepare_training_data.py) merges into labeled dataset |
| **Lag Features** | ✅ Complete | [src/data/feature_engineering.py](../src/data/feature_engineering.py) + auto-integration in [src/experiments/base.py](../src/experiments/base.py); tested in [tests/test_lag_feature_engineering.py](../tests/test_lag_feature_engineering.py) |
| **Seasonal Features** | ✅ Complete | [src/data/feature_engineering.py](../src/data/feature_engineering.py): month, day-of-year, season, growing-season, is_weekend |
| **Rolling Aggregates** | ✅ Complete | [src/data/feature_engineering.py](../src/data/feature_engineering.py): configurable windows & functions (mean, max, min, std) |
| **EDA & Visualization** | ✅ Complete | [src/visualization/plots.py](../src/visualization/plots.py), [src/cli/visualize.py](../src/cli/visualize.py) |
| **Models (Linear, Logistic, MLP)** | ✅ Complete | [src/models/](../src/models/): implementations + [src/experiments/](../src/experiments/) orchestration |
| **Training Loop** | ✅ Complete | [src/training/trainer.py](../src/training/trainer.py): epoch loop, checkpoint selection, batch inference |
| **Temporal Train/Val/Test Split** | ✅ Complete | [src/data/preprocessing.py](../src/data/preprocessing.py) `split_dataset()` supports temporal & custom splits |
| **Row-Level Metrics** | ✅ Complete | [src/events/evaluator.py](../src/events/evaluator.py): accuracy, precision, recall, F1, Brier score |
| **Event-Level Metrics** | ✅ Complete | [src/events/evaluator.py](../src/events/evaluator.py): onset error, event recall/precision, false alarm rate |
| **SHAP Explainability** | ✅ Complete | [src/explain/shap_explainer.py](../src/explain/shap_explainer.py) + [src/cli/explain.py](../src/cli/explain.py) |
| **Feature Importance Ranking** | ✅ Complete | [src/explain/feature_importance.py](../src/explain/feature_importance.py), markdown report generation |
| **CLI Interface** | ✅ Complete | [src/cli/](../src/cli/): fetch, preprocess, train, predict, evaluate, visualize, explain, report_summary, compare_experiments |
| **Experiment Registry** | ✅ Complete | [src/experiments/registry.py](../src/experiments/registry.py) w/ factory pattern |
| **Unit Tests** | ✅ Complete | [tests/](../tests/): domain features, lag features, event evaluation, preprocessing, trainer, experiments, netCDF loader, ERA5 processor — 113 passing |
| **Rule Schemas & Messages** | ✅ Complete | [src/events/rules.py](../src/events/rules.py): O1-O4 rules with thresholds and Polish messages |
| **Event Detection Rules (O1-O4)** | ✅ Complete | [src/events/detectors/](../src/events/detectors/): threshold logic for rainfall, thaw, seasonal rules |
| **Confidence Estimation** | ✅ Complete | Probabilities + historical frequency-based confidence calibration |
| **Statistical Analysis** | ✅ Complete | [src/analysis/](../src/analysis/): lag correlations, hypothesis tests, contingency tables, onset error distributions w/ Bonferroni/FDR corrections & normality checks |
| **PDF/DOCX Reporting** | ✅ Complete | [../reports/](../reports/) |
| **Notebook Pipeline** | ✅ Complete | [../notebooks/](../notebooks/seaData.ipynb) |
| **Continuous Forecast Pipeline** | ✅ Complete | `make continuous-predict`, `make api-server`, new endpoints [src/api/main.py](../src/api/main.py) |

---

## ✅ Fully Implemented

### 1. Data Pipeline (Steps 1–5 in checklist)

- **Fetch**: IMGW-PIB, ERA5, port/river data scripts exist.
- **Clean**: Missing-value imputation with dataset-specific strategies (valid-zero vs invalid-zero), text normalization, hydrological-to-calendar date conversion.
- **Sync & Resample**: Merge three water-level stations, validate hourly alignment, create daily aggregations (mean/max/min).
- **Tests**: [test_preprocessing.py](../tests/test_preprocessing.py), [test_lag_feature_engineering.py](../tests/test_lag_feature_engineering.py), [test_domain_feature_engineering.py](../tests/test_domain_feature_engineering.py).

### 2. Feature Engineering (Steps 6–8 in checklist)

All features implemented and tested:

- **Domain features**: rain sums (1/3/6/12/24h), temperature delta (24h), thaw flag (frozen→thaw transition), soil saturation (EWMA), wind components (U/V).
- **Lag features**: configurable 1–72h lags for rainfall, temperature, pressure; warmup row dropping; automatic column registration in preprocessing.
- **Seasonal features**: month, day-of-year, day-of-week, hour-of-day, season (winter/spring/summer/autumn), is_weekend, is_growing_season.
- **Rolling aggregates**: mean/max/min/std over configurable windows (3, 6, 12, 24h defaults).
- **Wiring**: Integrated into [src/experiments/base.py](../src/experiments/base.py) `preprocess()` method with config-driven toggles.

### 3. Training & Evaluation (Steps 9–11 in checklist, plus modeling)

- **Models**: Linear classifier, logistic regression (binary), MLP (multiclass/binary).
- **Training**: Epoch loop, Adam optimizer, checkpoint selection by best validation accuracy, binary/multiclass loss handling.
- **Evaluation**: Row-level metrics (accuracy, precision, recall, F1), event-level metrics (onset error in hours, event recall/precision, false-alarm rate, Brier score).
- **Temporal split**: Train (2021-2023), validation/test (configurable custom dates).
- **Tests**: [test_trainer.py](../tests/test_trainer.py), [test_experiments.py](../tests/test_experiments.py), [test_event_evaluator.py](../tests/test_event_evaluator.py).

### 4. Explainability & Interpretability

- **SHAP values**: Linear, deep (torch), and generic explainers.
- **Feature importance**: Mean absolute SHAP ranking, top-k selection.
- **Reporting**: Markdown report + CSV feature importance output.
- **CLI**: [src/cli/explain.py](../src/cli/explain.py) orchestrates SHAP computation, ranking, and report generation.

### 5. CLI & Experiment Orchestration

- **Commands**: fetch_data, preprocess_data, train_model, predict, evaluate_model, visualize, explain, report_summary, compare_experiments, run_experiment.
- **Factory pattern**: [src/experiments/registry.py](../src/experiments/registry.py) supports registration and lookup.
- **End-to-end pipeline**: `run_experiment` integrates all stages (preprocess → train → evaluate → visualize).

### 6. ERA5 Data Integration (NEW - May 2026)

- **NetCDF Loader**: [src/data/netcdf_loader.py](../src/data/netcdf_loader.py)
  - Handles `valid_time` coordinate recognition (ECMWF netCDF standard)
  - Spatial aggregation (2×2 Baltic grid → time series via mean)
  - 6-hourly → hourly resampling with linear interpolation
  - Deduplication on timestamp
- **ERA5 Processor**: [src/data/era5_processor.py](../src/data/era5_processor.py)
  - Wind extraction (u10/v10 m/s → wind_u, wind_v, wind_speed_ms, wind_direction_deg using meteorological convention)
  - Pressure extraction (msl Pa → pressure_hpa with unit conversion check)
  - SST extraction (K → C with coastal all-NaN handling for graceful degradation)
  - Yearly orchestration (2021-2025) with outer join merging
- **Training Data Integration**: [scripts/prepare_training_data.py](../scripts/prepare_training_data.py)
  - Auto-triggers ERA5 processor on first run
  - Merges ERA5 hourly CSV into labeled training dataset (left join on timestamp)
  - Handles duplicate pressure columns from weather + ERA5 (prefers ERA5)
  - Final output: `water_level_training_with_wind.csv` (43,790 rows, 12 columns)
- **Model Configs Updated**: [configs/linear_water_level.yaml](../configs/linear_water_level.yaml), [logistic_water_level.yaml](../configs/logistic_water_level.yaml), [mlp_water_level.yaml](../configs/mlp_water_level.yaml)
  - Now include: wind_u, wind_v, wind_speed, wind_direction, temp_delta_24h in feature_columns
  - All three models train successfully with new features
- **Unit Tests**: [tests/test_netcdf_loader.py](../tests/test_netcdf_loader.py), [tests/test_era5_processor.py](../tests/test_era5_processor.py)
  - 11 tests for netCDF loader (load, NaN handling, deduplication, resampling with interpolation, frame concatenation)
  - 8 tests for ERA5 processor (wind, pressure, SST extraction with unit conversion, all-NaN coastal handling, full-year orchestration)
  - All 19 tests passing with pandas 3.0+ compatibility fixes
- **Technical Achievements**:
  - ✅ Handles netCDF coordinate name variations (valid_time vs time vs timestamp)
  - ✅ Graceful fallback: models train successfully with/without ERA5 (no hard requirement)
  - ✅ Coastal SST edge case: correctly fills all-NaN with 0.0 during interpolation
  - ✅ Frequency code compatibility: pandas 3.0+ uses lowercase "h" not "H"
  - ✅ Type-safe: TYPE_CHECKING guards for xarray optional import
  - ✅ All code quality checks: ruff, pyright, pytest passing

### 7. Event Detection Rules (O1–O4)

**Status**: ✅ **Complete**

- **Rules defined**: [src/events/rules.py](../src/events/rules.py) with O1–O4 event types, thresholds, and Polish messages.
- **Detectors implemented**: [src/events/detectors/rainfall.py](../src/events/detectors/rainfall.py), [src/events/detectors/thaw.py](../src/events/detectors/thaw.py), [src/events/detectors/seasonal.py](../src/events/detectors/seasonal.py).
- **Logic**: Threshold-checking logic for rainfall (72h/7d), thaw (temp transition), and seasonal factors.

### 8. Confidence & Calibration

- **What exists**: Model outputs probabilities; Platt Scaling calibration in [src/training/calibration.py](../src/training/calibration.py); historical frequency confidence in [scripts/calculate_historical_confidence.py](../scripts/calculate_historical_confidence.py). Historical co-occurrence frequency (confidence as % of matching historical episodes that led to high water).
- **Relevant code**: [src/experiments/base.py](../src/experiments/base.py) lines 220–222 set confidence as `max(probabilities)`; [src/events/evaluator.py](../src/events/evaluator.py) computes Brier score.

### 9. Statistical Analysis

- **What exists**:
  - Lag correlation analysis (Pearson & Spearman) per season: [src/analysis/statistical_analyzer.py](../src/analysis/statistical_analyzer.py) `compute_lag_correlations()`
  - Hypothesis tests (t-test & Mann-Whitney U) for feature differences: `compare_groups_by_threshold()`
  - Soil saturation vs event contingency tables: `soil_saturation_event_crosstab()`
  - Onset error distribution analysis: `analyze_onset_error_distribution()`
  - Multiple testing corrections: Both Bonferroni and FDR; uncorrected p-values saved for comparison
  - Normality checks (Shapiro-Wilk) before group tests
  - Full orchestration: `StatisticalAnalyzer` class for end-to-end analysis
  - CLI command: [src/cli/analyze.py](../src/cli/analyze.py) for standalone statistical reports
  - Pipeline integration: Automatically computed in [src/experiments/base.py](../src/experiments/base.py) lines 287–301
  - 30 unit tests covering all functions: [tests/test_analysis.py](../tests/test_analysis.py)
- **Key features**:
  - Pair-wise deletion for missing values with warning generation
  - Data normalization validation before analysis
  - Markdown + JSON report generation
  - Per-season and all-data breakdowns
  - Effect size reporting (Cohen's d, rank-biserial, Cramér's V)

### 10. Automated PDF/DOCX Report Generation

- **What exists**: [src/reports/report_generator.py](../src/reports/report_generator.py) generates PDF reports with narrative sections and figures (seaSON plots, correlation heatmaps, event timelines). CSV outputs for structured factor tables (season, threshold, lag, confidence per O1–O4) are generated in [src/cli/report_summary.py](../src/cli/report_summary.py).

### 11. Consolidated Notebook & User Guide

- **Notebook**: [notebooks/seaData_pipeline.ipynb](../notebooks/seaData_pipeline.ipynb) provides an end-to-end walkthrough of the data pipeline, modeling, evaluation, and analysis with explanations and visualizations.
- **User Guide**: [docs/USAGE.md](docs/USAGE.md) contains common commands, interpretation guide, and troubleshooting tips for users.

### 12. Continuous Evaluation & Forecasting Pipeline

- **Continuous Prediction**: [src/continuous/service.py](../src/continuous/service.py) Orchestrates fetching data from IMGW, Open-Meteo, and Stormglass, running horizon predictions (+1d, +3d, +7d), and generating SHAP values for each target day. Runs from a single zero-parameter command `make continuous-predict`.
- **Thaw Detector Fix**: Event detectors strictly require freezing temperatures (`min_previous <= 0.0`) and temperature thresholds, avoiding false thaw alerts during warm seasons (such as mid-May).
- **FastAPI Endpoints**: Exposes endpoints tailored for frontend consumption:
  - `GET /continuous/forecast` - Aggregated dashboard of current + horizon predictions.
  - `GET /continuous/status` - Freshness & health check for the connected APIs and latest runs.
  - `GET /continuous/predictions/{horizon}` - Detailed telemetry for specific forecast horizons.
- **Rich Visualization**: Generates a 3-panel visual plot (`continuous_forecast_YYYY-MM-DD.png`) detailing risk vs. confidence, dual-axis weather (rainfall + temperature), and barometric pressure alongside warnings.

---

## ⚠️ Partially Implemented

---

## ❌ Not Yet Implemented

---

## 🎯 Prioritized Implementation Backlog

### Phase 1: Event Detection Completion (High Impact)

1. **Implement detector bodies** ([src/events/detectors/](../src/events/detectors/)): threshold logic for rainfall, thaw, seasonal.
2. **Wire detectors into evaluation**: integrate rule detections into experiment reports alongside model predictions.
3. **Test detectors**: add unit tests comparing rule outputs to model predictions.

### Phase 2: Confidence & Interpretability (High Impact)

1. **Calibration**: add temperature scaling or isotonic regression to model probabilities.
2. **Event confidence**: compute per-event confidence as historical co-occurrence frequency or posterior probability.
3. **Factor attribution**: extend SHAP reports to map top features to O1–O4 categories.
4. **Update predictions CSV**: add event type, confidence, contributing factors.

### Phase 3: Statistical Analysis & Validation (Medium Impact)

1. **Seasonal hypothesis tests**: t-tests/Mann-Whitney for rainfall/temperature differences in high vs low water seasons.
2. **Lag sensitivity analysis**: correlation per season by lag, heatmaps.
3. **Cross-validation**: K-fold or temporal walk-forward validation; onset error distribution.
4. **Reports**: summarize metrics per season and year; save to [reports/global/](../reports/global/).

### Phase 4: Reporting & Documentation (High Impact)

1. **PDF generation**: add narrative sections + figures (seaSON plots, correlations, events) to PDF.
2. **CSV outputs**: structured factor tables (season, threshold, lag, confidence per O1–O4).
3. **Notebook walkthrough**: create or update [notebooks/seaData_pipeline.ipynb](../notebooks/seaData_pipeline.ipynb) with end-to-end example.
4. **User guide**: [docs/USAGE.md](docs/USAGE.md) with common commands, interpretation guide, troubleshooting.

---

## Quick Commands

```bash
# Run tests
pytest -q

# Run full pipeline
python -m src.cli.run_experiment configs/linear_water_level.yaml

# Preprocess only
python -m src.cli.preprocess_data configs/linear_water_level.yaml

# Train only
python -m src.cli.train_model configs/linear_water_level.yaml

# Evaluate only
python -m src.cli.evaluate_model configs/linear_water_level.yaml

# Explain model
python -m src.cli.explain <config> <model_checkpoint> <data_csv>

# Summarize reports
python -m src.cli.report_summary --reports-root reports --output-csv reports/global/experiment_summary.csv

# Compare experiments
python -m src.cli.compare_experiments configs/compare_all_models.yaml

# Run continuous forecast prediction (zero-parameter)
make continuous-predict

# Start REST API server
make api-server
```

---

## Notes for Contributors

1. **Feature toggles**: Lag, seasonal, and rolling features are config-driven in [src/utils/config.py](../src/utils/config.py). Toggle them in the YAML config file under `feature_engineering` section.
2. **Event detection**: Rule thresholds are in [src/events/rules.py](../src/events/rules.py); detector logic stubs are in [src/events/detectors/](../src/events/detectors/). Start by implementing `detect_*()` functions with actual threshold checks.
3. **Experiments**: Register new model types in [src/experiments/registry.py](../src/experiments/registry.py).
4. **Tests**: Always add tests for new features (see [tests/](../tests/) for patterns).
5. **Backward compatibility**: Ensure config changes are backward-compatible with existing YAML files.
