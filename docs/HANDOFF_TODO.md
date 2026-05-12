# Handoff: Remaining Implementation Tasks

**Last Updated**: May 12, 2026  
**Status**: 113/126 test cases passing • All linting (ruff) & type checking (pyright) passing • Full data pipeline operational

---

## Executive Summary

The project has completed:

- ✅ Full data pipeline (fetch → clean → sync → resample → aggregate)
- ✅ Feature engineering (domain, lag, seasonal, rolling + ERA5 wind/pressure/SST)
- ✅ Model training (Linear, Logistic, MLP with SHAP explainability)
- ✅ Statistical analysis (correlations, hypothesis tests, onset error analysis)
- ✅ CLI interface (8 commands covering all pipeline stages)
- ✅ Unit tests (113 passing across all modules)
- ✅ ERA5 netCDF data integration with graceful fallback

**Remaining work** is primarily:

1. **Event detection logic** (O1-O4 rule implementation) — MEDIUM effort
2. **Confidence calibration** (model probability adjustment) — SMALL effort
3. **PDF reporting** (automated narrative + figures) — MEDIUM effort
4. **Notebook & documentation** (user guide, walkthrough) — SMALL effort

**Estimated effort**: 2-4 weeks for comprehensive completion, 1 week for MVP.

---

## 📋 Detailed Remaining Work

### 1. Event Detection Implementation (MEDIUM Effort)

**Status**: Rule schemas exist; detection logic is placeholder (always returns `detected=False`).

**Location**: `src/events/detectors/`

#### 1.1 Rainfall Detector - Flash Flood (O1)

- **File**: `src/events/detectors/rainfall.py`
- **Function**: `detect_flash_flood()`
- **Task**: Implement threshold logic
  - Check 6h cumulative rainfall > 90th percentile (historical)
  - Return `EventDetection(detected=True, confidence=0.85, threshold=<value>)`
  - Wire into rule response message (already defined: "Wysoka intensywność opadu...")
- **Test location**: [tests/test_event_detector.py](../tests/) — **create new test file**
- **Integration**: Update `src/events/rules.py` to call actual detector instead of placeholder

#### 1.2 Long Rainfall Detector (O2)

- **File**: `src/events/detectors/rainfall.py`
- **Function**: `detect_long_rainfall()` (or extend `detect_flash_flood()`)
- **Task**: Implement cumulative rainfall logic
  - Check 72h + 7d rainfall > seasonal threshold
  - Check soil saturation indicator (EWMA in preprocessing)
  - Return confidence based on soil saturation level
- **Integration**: Ensure message reflects both rainfall & saturation context

#### 1.3 Thaw Detector (O3)

- **File**: `src/events/detectors/thaw.py`
- **Function**: `detect_thaw()`
- **Task**: Implement temperature logic
  - Check temp > 0°C for 24+ hours
  - Check lookback window (7d) had sub-zero temperatures (frozen condition)
  - Optional: Integrate wind context (high wind increases evaporation)
  - Return confidence proportional to temperature stability
- **Enhancement idea**: Add wind_speed context
  - `get_wind_direction_category(wind_u, wind_v)` — compute cardinal direction
  - If wind points downstream (toward flood risk area), increase severity
- **Test**: Compare detection against ground-truth thaw events in data

#### 1.4 Seasonal Dependency Detector (O4)

- **File**: `src/events/detectors/seasonal.py`
- **Function**: `detect_seasonal_dependencies()`
- **Task**: Analyze seasonal patterns
  - Compute dominant feature per season (rainfall, temperature, wind)
  - Check if current conditions match dominant seasonal driver
  - Return confidence as historical frequency (% of season's high-water events driven by this factor)
- **Complexity**: Requires historical event grouping (rainfall-dominated vs thaw-dominated vs wind-driven vs other)

#### 1.5 Integration Points

- **File**: `src/events/rules.py`
  - Wire detector function calls into rule evaluation
  - Currently: `rule.detector(...)` returns placeholder; implement actual call logic
- **File**: `src/events/evaluator.py`
  - Ensure detector outputs are included in event reports (currently model predictions only)
  - Add columns: `detection_method`, `detection_confidence`, `contributing_factors`
- **File**: `scripts/prepare_training_data.py`
  - No changes needed; detectors use pre-computed features (rainfall_mm, temperature_c, etc.)

---

### 2. Confidence Calibration (SMALL Effort)

**Status**: Model outputs probabilities; all predictions use `confidence = max(probabilities)`. No calibration applied.

**Location**: `src/models/` and `src/experiments/base.py`

#### 2.1 Temperature Scaling Calibration

- **What it is**: Linear post-hoc adjustment to model confidence scores
- **Where to add**: `src/models/base.py` or new file `src/models/calibration.py`
- **Implementation**:

  ```python
  def calibrate_temperature_scaling(logits, val_labels, test_logits):
      # Fit temperature T on validation set, apply to test
      # Minimizes negative log-likelihood on validation set
  ```

- **Integration**: Apply after model prediction in `src/experiments/base.py` line 220
- **Test**: Compare Brier score before/after calibration ([src/events/evaluator.py](../src/events/evaluator.py) line 265 already computes it)

#### 2.2 Per-Event Confidence (Historical Co-occurrence)

- **What it is**: Confidence = % of matching rule conditions that historically led to high water
- **Where to compute**: `src/events/evaluator.py` in event matching phase
- **Algorithm**:
  1. Group historical events by O1-O4 rule type
  2. Count how many matched high-water labels (true positives)
  3. Compute confidence = TP / (TP + FP) for each rule type
  4. Store as lookup: `{rule_type: confidence}`
  5. Apply when detector fires
- **Integration**: Update detector return to include this historical confidence

#### 2.3 CSV Output Enhancement

- **File**: `src/experiments/base.py` line 225 (predictions_frame creation)
- **Add columns**:
  - `event_type` (O1/O2/O3/O4 or "none")
  - `detection_confidence` (from detector)
  - `model_confidence` (max probability, post-calibration)
  - `ensemble_confidence` (average of detection + model confidence)
  - `contributing_factors` (top SHAP features for this prediction)

---

### 3. Wind-Context Event Detection Enhancement (OPTIONAL)

**Status**: Wind features (wind_u, wind_v, wind_speed_ms, wind_direction_deg) are available in training data. Can enhance detectors.

**Location**: `src/events/detectors/rainfall.py`, `src/events/detectors/thaw.py`

#### 3.1 Wind-Driven Transport (Rainfall)

- **Idea**: High wind + rainfall direction pointing toward study area → higher flood risk
- **Implementation**:
  - Compute wind direction: `atan2(wind_u, wind_v) * 180/pi + 360 mod 360` (already in ERA5 processor)
  - Define "flood-risk directions" (e.g., SW, W, NW for Baltic coast)
  - Increase severity if wind_speed > 5 m/s AND direction in risk zone
  - Decrease severity if wind blows away from study area (transport effect reduces local rainfall impact)
- **Function**: Add `get_wind_direction_category()` helper

#### 3.2 Wind-Driven Evaporation (Thaw)

- **Idea**: High wind during thaw → faster snow melting and runoff
- **Implementation**:
  - If thaw condition detected AND wind_speed > 6 m/s, increase confidence
  - Log wind context in rule response message
- **Function**: Extend `detect_thaw()` to include wind data check

---

### 4. PDF Report Generation (MEDIUM Effort)

**Status**: Current output is markdown + CSV. No PDF generation.

**Location**: `src/reporting/` (new module) and `src/cli/`

#### 4.1 Setup & Tools

- **Existing dependency**: `reportlab` (already in pyproject.toml)
- **Alternative**: `python-docx` for DOCX format (also in dependencies)
- **Choice**: `reportlab` for PDF (more control; smaller file size)

#### 4.2 Report Structure

```python
[PDF Report]
├── Title Page
│   ├── Project name, date, model name
│   └── Executive summary (3-5 lines)
├── Methodology Section
│   ├── Data sources (IMGW-PIB, ERA5, ports)
│   ├── Features used (list + brief description)
│   ├── Model architecture & training strategy
│   └── Evaluation metrics (temporal split, onset error definition)
├── Results Section
│   ├── Feature Importance (SHAP top-10 bar chart)
│   ├── Seasonality (water level time series by season)
│   ├── Correlation Analysis (heatmap: features vs water level, per season)
│   ├── Event Detection Results
│   │   ├── Overall metrics (event recall, precision, false alarm rate)
│   │   ├── Per-season breakdown (table)
│   │   └── Example detected events (5-10 recent examples with dates, confidence)
│   └── Thresholds & Factors (per-season table: O1-O4 criteria, thresholds, confidence)
├── Interpretability Section
│   ├── Top contributing factors (SHAP values for example predictions)
│   ├── Risk communication (example: "What triggers a high-water alert?")
│   └── Limitations (model assumptions, edge cases, coastal SST caveat)
└── Appendix
    └── Glossary (Polish terms, technical definitions)
```

#### 4.3 Implementation Files

- **New file**: `src/reporting/pdf_generator.py`

  ```python
  class PDFReporter:
      def __init__(self, config, results_dict):
          self.config = config
          self.results = results_dict  # from experiment run
      
      def generate(self, output_path):
          doc = SimpleDocTemplate(output_path, ...)
          story = []
          story.append(self._title_page())
          story.append(self._methodology_section())
          story.append(self._results_section())
          story.append(self._interpretability_section())
          doc.build(story)
  ```

- **Integration**: Call from `src/cli/run_experiment.py` after evaluation completes (around line 80)

#### 4.4 Figure Generation

- **Seasonality plot**: Already exists in `src/visualization/plots.py` → export to PNG
- **SHAP plots**: Already computed in [src/explain/shap_explainer.py](../src/explain/shap_explainer.py) → call `matplotlib.pyplot.savefig()`
- **Correlation heatmap**: Create new function in `src/visualization/plots.py`:

  ```python
  def plot_correlation_heatmap(df, output_path):
      # Features vs water_level, colored by season
  ```

- **Event table**: Generate from `src/events/evaluator.py` event list

#### 4.5 Testing

- **Create**: `tests/test_pdf_generator.py`
- **Smoke test**: Generate PDF with dummy data; verify file exists and is readable

---

### 5. Notebook & Documentation (SMALL Effort)

**Status**: CLI exists; no consolidated notebook or user guide.

**Location**: `notebooks/` and `docs/`

#### 5.1 Jupyter Notebook Walkthrough

- **File**: `notebooks/seaData_pipeline.ipynb` (create or fully rewrite)
- **Cells**:
  1. Setup & imports
  2. Load preprocessed CSV (or run preprocessing)
  3. Train model (or load checkpoint)
  4. Evaluate model & show metrics
  5. Visualize seasonality
  6. Generate SHAP explanations
  7. Detect events (O1-O4)
  8. Generate PDF report
  9. Interpretation guide (what do these results mean?)
- **Execution time**: ~2-3 minutes on sample data
- **Target audience**: Data scientist or project manager wanting to understand the pipeline

#### 5.2 USAGE.md User Guide

- **File**: `docs/USAGE.md` (create new)
- **Sections**:
  1. **Quick Start**
     - Installation (venv, dependencies)
     - Run full pipeline: `make run`
     - View results in `reports/`
  2. **Available Commands**
     - fetch_data, preprocess_data, train_model, predict, evaluate_model, visualize, explain, compare_experiments
     - Example: `python -m src.cli.preprocess_data configs/linear_water_level.yaml`
  3. **Configuration**
     - How to edit YAML configs
     - Feature toggles (lag, seasonal, rolling)
     - Model selection & hyperparameters
  4. **Output Interpretation**
     - Metrics explained (event recall vs precision)
     - SHAP values and feature importance
     - Confidence scores and risk communication
  5. **Troubleshooting**
     - Common errors & solutions
     - How to check data quality
     - How to validate feature engineering
  6. **Advanced Topics**
     - Custom detectors (implementing O1-O4 logic)
     - Probability calibration
     - Temporal validation strategies

#### 5.3 README Update

- **File**: `README.md` (update existing)
- **Add**: Links to USAGE.md and new notebook
- **Add**: Project status matrix (summary of completed vs pending work)

#### 5.4 README in notebooks/

- **File**: `notebooks/README.md` (create new)
- **Content**:
  - Notebook title & purpose
  - Prerequisites (what's needed before running)
  - How to run (kernel selection, data path)
  - Output interpretation

---

### 6. Optional Enhancements (NOT REQUIRED FOR MVP)

These are nice-to-haves; implement if time permits.

#### 6.1 Cross-Validation & Walk-Forward Validation

- **Idea**: Current split is train (2021-23) vs test (2024-25). Add k-fold or rolling-window validation.
- **Location**: `src/data/preprocessing.py` `split_dataset()` enhancement
- **Benefit**: More robust onset error estimates

#### 6.2 Seasonal Hypothesis Testing

- **Idea**: Test if rainfall/temperature effects differ by season (statistical significance)
- **Location**: Extend `src/analysis/statistical_analyzer.py`
- **Test**: Interaction terms: does rainfall's effect on water level change with season?

#### 6.3 Hyperparameter Optimization

- **Idea**: Grid search or Bayesian optimization for model hyperparameters (learning rate, layer sizes, etc.)
- **Tool**: `optuna` or `hyperopt`
- **Location**: New file `src/experiments/hyperparameter_tuning.py`

#### 6.4 Multi-Location Support

- **Idea**: Current pipeline is single-location (Baltic). Extend to multiple rivers/stations.
- **Effort**: Significant (config refactoring, data loading)
- **Benefit**: Reusability for other hydrological domains

#### 6.5 Real-Time Prediction API

- **Idea**: FastAPI server for live predictions (given current weather, predict flood risk)
- **Tool**: FastAPI, Pydantic
- **Endpoints**: `/predict`, `/explain`, `/historical-events`

---

## 🚀 Recommended Implementation Order

### MVP (1 week)

1. ✅ **Event Detection Logic** (rainfall + thaw detectors)
2. ✅ **Confidence Calibration** (temperature scaling)
3. ✅ **Notebook Walkthrough** (4-5 key cells)

### Phase 2 (1 week)

1. PDF report generation (basic structure + figures)
2. Wind context enhancement (optional: add to detectors)
3. USAGE.md documentation

### Phase 3 (Optional)

1. Seasonal hypothesis testing
2. Cross-validation
3. Real-time API

---

## 📊 Remaining Test Coverage

**Current**: 113 passing tests  
**TODO**:

- [ ] Event detector unit tests (rainfall, thaw, seasonal): **3-5 new test classes**
- [ ] PDF generator smoke tests: **1 new test class**
- [ ] Confidence calibration tests: **1-2 new test classes**

**Target**: 130+ total tests (17 new tests)

---

## 🔗 Key File Reference

| Task | Primary Files | Secondary Files |
| ------ | --- | --- |
| Event Detection | `src/events/detectors/` | `src/events/rules.py`, `src/events/evaluator.py` |
| Confidence Calibration | `src/models/calibration.py` (new) | `src/experiments/base.py` (update line 220) |
| PDF Reporting | `src/reporting/pdf_generator.py` (new) | `src/visualization/plots.py` (enhance) |
| Documentation | `docs/USAGE.md` (new), `notebooks/seaData_pipeline.ipynb` | `README.md` (update) |
| Wind Context | `src/events/detectors/rainfall.py`, `thaw.py` | `src/data/era5_processor.py` (already has wind_direction) |

---

## 📝 Notes for Implementer

1. **Backward Compatibility**: Ensure config YAML changes are backward-compatible. Old configs should still work (or fail gracefully).

2. **Testing Strategy**: Each new function should have corresponding unit test. Use pytest fixtures for reusable test data.

3. **Documentation**: Add docstrings (Google style) to all new functions. Update [docs/IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) as you complete tasks.

4. **ERA5 Integration**: Wind features are now in training CSV (`wind_u`, `wind_v`, `wind_speed_ms`, `wind_direction_deg`). Use them in detectors and analysis.

5. **Graceful Degradation**: If ERA5 processor fails, training still uses weather-only data (existing behavior preserved).

6. **CI/CD**: All checks must pass:

   ```bash
   make ruff      # linting
   make pyright   # type checking
   make test      # unit tests (113 currently)
   make checks    # all three above
   ```

---

## 🎯 Success Criteria

- [ ] All 4 event detectors (O1-O4) implemented with threshold logic
- [ ] 20+ unit tests for detectors (TDD approach)
- [ ] Confidence calibration integrated & tested
- [ ] PDF report generator produces 10+ page document with figures
- [ ] Notebook runs end-to-end in <5 minutes
- [ ] USAGE.md covers all CLI commands with examples
- [ ] All 130+ tests passing
- [ ] Ruff, pyright, pytest all passing
- [ ] Code coverage >80% for new modules

---

## 📞 Questions & Clarifications

**Q: Should detectors use hardcoded thresholds or learn from data?**  
A: Recommendation: Start with historical percentiles (90th-95th), computed per season. Later, optimize via grid search.

**Q: How to handle missing ERA5 data (coastal SST issue)?**  
A: Already handled! Fills with 0.0 during training prep. Detectors should note this in confidence (don't increase confidence for SST-based decisions if SST is all-NaN).

**Q: Should event detection run alongside or after model predictions?**  
A: Both. Rule-based detection (O1-O4) is independent; model provides ML-based predictions. Combine both in final report (ensemble approach).

**Q: What if seasonal hypothesis tests show no significant differences?**  
A: Still document the finding. It means rainfall effects are season-invariant (or model isn't sensitive enough). Either way, it's valuable insight.

---

**For quick updates on progress, check the bottom of this file or refer to [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) status matrix.**
