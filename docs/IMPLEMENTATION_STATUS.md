# Implementation Status - seaData/backend

Comprehensive audit of the project pipeline against [project_sheet.md](project_sheet.md) and [checklist.md](checklist.md) deliverables. Status reflects evidence from source code, tests, and CLI interfaces as of May 2026.

---

## Status Summary

| Phase | Status | Evidence |
| ------- | -------- | ---------- |
| **Data Ingestion** | ✅ Complete | [scripts/fetch_imgw_dataset.py](../scripts/fetch_imgw_dataset.py), [data/fetch_era5_data.py](../data/fetch_era5_data.py) |
| **Data Cleaning** | ✅ Complete | [src/data/preprocessing.py](../src/data/preprocessing.py) w/ missing-value strategies, normalization, and renaming |
| **Synchronization & Resampling** | ✅ Complete | [src/data/synchronization.py](../src/data/synchronization.py) w/ merge, alignment validation, daily aggregations |
| **Domain Features (O1-O4 inputs)** | ✅ Complete | [src/data/feature_engineering.py](../src/data/feature_engineering.py): rain sums, temp delta, thaw flag, soil saturation, wind components |
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
| **Unit Tests** | ✅ Complete | [tests/](../tests/): domain features, lag features, event evaluation, preprocessing, trainer, experiments |
| **Rule Schemas & Messages** | ✅ Complete | [src/events/rules.py](../src/events/rules.py): O1-O4 rules with thresholds and Polish messages |
| **Event Detection Placeholders** | ⚠️ Partial | [src/events/detectors/](../src/events/detectors/): schemas exist, response messages defined, **detection logic is placeholder (returns `detected=False`)** |
| **Confidence Estimation** | ⚠️ Partial | Probabilities + SHAP tooling available; **event-level confidence calibration not implemented** |
| **Statistical Analysis** | ⚠️ Partial | Correlation/Brier score computed; **detailed seasonal breakdown & thaw/rainfall hypothesis tests missing** |
| **PDF/DOCX Reporting** | ❌ Not Implemented | Current output is markdown + CSV; **no automated PDF generation** |
| **Notebook Pipeline** | ❌ Not Implemented | CLI commands exist; **no consolidated .ipynb walkthrough or user guide doc** |

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

---

## ⚠️ Partially Implemented

### 1. Event Detection Rules (O1–O4)

**Status**: Rule schemas and response messages **complete**; detection logic **placeholder**.

- **Rules defined**: [src/events/rules.py](../src/events/rules.py) with O1–O4 event types, thresholds, and Polish messages.
- **Detectors stubbed**: [src/events/detectors/rainfall.py](../src/events/detectors/rainfall.py), [src/events/detectors/thaw.py](../src/events/detectors/thaw.py), [src/events/detectors/seasonal.py](../src/events/detectors/seasonal.py).
- **Missing**: Actual threshold-checking logic; all detectors currently return `detected=False` with placeholder `confidence=0.82`.
- **Needed to complete**:
  - Rainfall: check 72h/7d cumulative rainfall vs seasonal thresholds.
  - Thaw: check temperature > 0°C + sub-zero in lookback window.
  - Seasonal: analyze dominant factors per season.

### 2. Confidence & Calibration

**Status**: Basic probabilities + SHAP available; event-level calibration **missing**.

- **What exists**: Model outputs probabilities; predictions_frame includes `confidence` (max probability); Brier score computed.
- **What's missing**:
  - Confidence calibration (temperature scaling, isotonic regression).
  - Per-event confidence scores tied to O1–O4 rules.
  - Historical co-occurrence frequency (confidence as % of matching historical episodes that led to high water).
- **Relevant code**: [src/experiments/base.py](../src/experiments/base.py) lines 220–222 set confidence as `max(probabilities)`; [src/events/evaluator.py](../src/events/evaluator.py) computes Brier score.

### 3. Statistical Analysis

**Status**: Basic metrics computed; detailed seasonal breakdown **partial**.

- **What exists**: Correlation, Brier score, event metrics computed per binary evaluation; `add_temporal_columns()` and `summarize_by_period()` support seasonal breakdowns.
- **What's missing**:
  - Explicit seasonal hypothesis tests (e.g., rainfall vs temperature contribution per season).
  - Lag correlation analysis showing which lags are strongest per season.
  - Soil saturation vs event occurrence cross-tabulation.
- **Relevant code**: [src/events/evaluator.py](../src/events/evaluator.py) lines 306–355 (summarize_by_period) partially support this.

---

## ❌ Not Yet Implemented

### 1. Automated PDF/DOCX Report Generation

**Status**: Not started.

- **Current output**: Markdown explainability report + CSV summaries.
- **Project sheet requirement**: PDF or DOCX report with sezonowość (seasonality), meteo→water-level relationships, event interpretation, and factor tables per season.
- **Tools available**: `reportlab` (PDF), `python-docx` (DOCX) can be added.
- **Scope**: Figure exports (seasonality plots, correlation heatmaps), narrative sections, threshold/factor tables.

### 2. Consolidated Notebook & User Guide

**Status**: Not started.

- **Current state**: CLI commands exist; [notebooks/seaData_pipeline.ipynb](../notebooks/seaData_pipeline.ipynb) exists but unclear if current.
- **Project sheet requirement**: Reproducible Jupyter notebook showing the full pipeline: data prep → feature engineering → event detection → risk communication.
- **Needed**:
  - Walkthrough with example data.
  - Output interpretation guide.
  - README in [notebooks/](../notebooks/) explaining how to run the notebook.
- **Quick start docs**: No [docs/USAGE.md](docs/USAGE.md) yet.

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
python -m src.cli.run_experiment configs/EXAMPLE_linear_minmax.yaml

# Preprocess only
python -m src.cli.preprocess_data configs/EXAMPLE_linear_minmax.yaml

# Train only
python -m src.cli.train_model configs/EXAMPLE_linear_minmax.yaml

# Evaluate only
python -m src.cli.evaluate_model configs/EXAMPLE_linear_minmax.yaml

# Explain model
python -m src.cli.explain <config> <model_checkpoint> <data_csv>

# Summarize reports
python -m src.cli.report_summary --reports-root reports --output-csv reports/global/experiment_summary.csv

# Compare experiments
python -m src.cli.compare_experiments configs/EXAMPLE_compare_experiments.yaml
```

---

## Notes for Contributors

1. **Feature toggles**: Lag, seasonal, and rolling features are config-driven in [src/utils/config.py](../src/utils/config.py). Toggle them in the YAML config file under `feature_engineering` section.
2. **Event detection**: Rule thresholds are in [src/events/rules.py](../src/events/rules.py); detector logic stubs are in [src/events/detectors/](../src/events/detectors/). Start by implementing `detect_*()` functions with actual threshold checks.
3. **Experiments**: Register new model types in [src/experiments/registry.py](../src/experiments/registry.py).
4. **Tests**: Always add tests for new features (see [tests/](../tests/) for patterns).
5. **Backward compatibility**: Ensure config changes are backward-compatible with existing YAML files.
