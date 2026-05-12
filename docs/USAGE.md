# Usage Guide — seaData Pipeline

Quick reference for running the water-level prediction pipeline. For detailed implementation status, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

---

## Setup

### Prerequisites

- Python 3.10+
- Virtual environment (recommended: `.venv/`)
- Dependencies: `pip install -r requirements.txt` (or as defined in `pyproject.toml`)

### Environment

```bash
cd /root/seaData/backend
source .venv/bin/activate  # or: conda activate seadata
```

---

## Quick Start: Full Pipeline

Run all stages (preprocess → train → evaluate → visualize) for a single experiment:

```bash
python -m src.cli.run_experiment configs/EXAMPLE_linear_minmax.yaml
```

Output location: `reports/<experiment_name>/` (train/val/test splits, model checkpoint, metrics, plots).

---

## Step-by-Step Pipeline

### 1. **Fetch & Load Raw Data**

Load raw data from CSV or ERA5:

```bash
python -m src.cli.fetch_data configs/EXAMPLE_linear_minmax.yaml \
  --source-csv data/raw/dead-vistula-river-water-level-2021-2025.csv
```

Or with config-specified source (if dataset path is in config).

### 2. **Preprocess Data**

Clean, handle missing values, generate features (lag, seasonal, rolling), and split into train/val/test:

```bash
python -m src.cli.preprocess_data configs/EXAMPLE_linear_minmax.yaml
```

**Output:**

- `data/processed/train.csv`, `validation.csv`, `test.csv`
- `models/<experiment_name>/preprocessor.json` (feature statistics for scaling)

**Config options** (in YAML under `feature_engineering`):

- `generate_lag_features: true` — Generate 1–72h lagged rainfall, temperature, pressure.
- `lag_hours: 72` — Maximum lag window (hours).
- `generate_seasonal_features: true` — Month, season, day-of-week, etc.
- `generate_rolling_aggregates: true` — Mean/max/min/std over 3/6/12/24h windows.

### 3. **Train Model**

Train the model specified in config (linear_classifier, logistic_regression, or mlp):

```bash
python -m src.cli.train_model configs/EXAMPLE_linear_minmax.yaml
```

**Output:**

- `models/<experiment_name>/checkpoint.pt` (best model by validation accuracy)
- `reports/<experiment_name>/training_history.csv` (epoch loss/accuracy)
- `reports/<experiment_name>/training_summary.json` (final metrics)

### 4. **Evaluate Model**

Evaluate on test set and compute row-level + event-level metrics:

```bash
python -m src.cli.evaluate_model configs/EXAMPLE_linear_minmax.yaml
```

**Output:**

- `reports/<experiment_name>/predictions.csv` (with predicted_class, confidence, probabilities)
- `reports/<experiment_name>/evaluation.json` (accuracy, event metrics, seasonal breakdowns)

**Metrics included:**

- Row-level: accuracy, precision, recall, F1, Brier score
- Event-level: onset error (hours), event recall/precision, false alarm rate
- Seasonal breakdown: metrics per year and per season (winter/spring/summer/autumn)

### 5. **Visualize Results**

Generate training curves and confusion matrix:

```bash
python -m src.cli.visualize configs/EXAMPLE_linear_minmax.yaml
```

**Output:**

- `reports/<experiment_name>/training_curves.png` (loss & accuracy over epochs)
- `reports/<experiment_name>/confusion_matrix.png`

### 6. **Explain Model (SHAP)**

Compute SHAP values, rank features by importance, and generate report:

```bash
python -m src.cli.explain <config> <model_checkpoint> <data_csv>
```

Example:

```bash
python -m src.cli.explain \
  configs/EXAMPLE_linear_minmax.yaml \
  models/my_experiment/checkpoint.pt \
  data/processed/test.csv
```

**Output:**

- `reports/<experiment_name>/feature_importance.csv` (ranked features with importance scores)
- `reports/<experiment_name>/explainability_report.md` (top 10 features + full table)
- `reports/<experiment_name>/shap_summary.png` (SHAP beeswarm plot)
- `reports/<experiment_name>/shap_bar.png` (feature importance bar chart)

### 7. **Summarize Experiments**

Aggregate evaluation JSON files from multiple runs into one summary CSV:

```bash
python -m src.cli.report_summary \
  --reports-root reports \
  --output-csv reports/global/experiment_summary.csv
```

Output: Top performers ranked by accuracy.

### 8. **Compare Experiments**

Compare metrics across multiple configs:

```bash
python -m src.cli.compare_experiments configs/EXAMPLE_compare_experiments.yaml
```

Generates comparison tables and bar charts.

---

## Single Prediction 1

### 8. **Statistical Analysis**

Perform comprehensive statistical analysis on predictions: lag correlations, hypothesis tests, soil saturation contingency, onset error distributions.

```bash
python -m src.cli.analyze reports/<experiment_name>/predictions.csv \
  --output-md reports/<experiment_name>/statistical_analysis.md \
  --output-json reports/<experiment_name>/statistical_analysis.json
```

**Key features:**

- **Lag Correlations**: Pearson & Spearman correlations between lag features (rainfall_lag_1h, rainfall_lag_6h, etc.) and water level, per season
- **Hypothesis Tests**: Compare feature values (rainfall, temperature, pressure, soil saturation) between high and low water level groups using:
  - **t-test** (parametric, assumes normality)
  - **Mann-Whitney U** (non-parametric, robust to outliers)
- **Multiple Testing Corrections**: Both uncorrected and corrected p-values:
  - **Bonferroni**: Conservative, controls family-wise error rate
  - **FDR** (Benjamini-Hochberg): Less conservative, controls false discovery rate
- **Normality Tests**: Shapiro-Wilk test for each group before selecting test type
- **Soil Saturation Contingency**: Chi-square test comparing soil saturation levels (quartiles) vs event occurrence
- **Onset Error Distribution**: Statistical summary of onset errors (predicted vs actual event start) per season
- **Effect Sizes**: Cohen's d (t-test), rank-biserial (Mann-Whitney), Cramér's V (chi-square)

**Output:**

- `statistical_analysis.md` — Markdown report with tables and interpretation
- `statistical_analysis.json` — JSON summary for integration with other tools

**Example with custom options:**

```bash
python -m src.cli.analyze reports/my_experiment/predictions.csv \
  --target-column water_level_m \
  --event-column event_occurred \
  --threshold-percentile 95 \
  --dataset-name "Vistula_2024" \
  --output-md /tmp/stats_report.md \
  --output-json /tmp/stats_summary.json
```

**Interpreting Results:**

1. **Lag Correlations**: Look for lags with |r| > 0.3 and p-value < 0.05. Strongest correlations suggest optimal lag times for prediction.
2. **Hypothesis Tests**:
   - If Shapiro-Wilk p > 0.05 for both groups: data is likely normal, t-test is reliable.
   - If p < 0.05: data is non-normal, prefer Mann-Whitney U result.
   - If Bonferroni/FDR p-value < 0.05: result is statistically significant after multiple testing correction.
3. **Soil Saturation**: If chi-square p < 0.05 and Cramér's V > 0.2, soil saturation significantly affects event occurrence.
4. **Onset Errors**: Compare median/IQR across seasons. Large seasonal differences suggest mechanistic differences.

### 9. **Compare Experiments**

Compare metrics across multiple configs:

```bash
python -m src.cli.compare_experiments configs/EXAMPLE_compare_experiments.yaml
```

Generates comparison tables and bar charts.

---

## Single Prediction 2

Predict water level for one instance:

```bash
python -m src.cli.predict configs/EXAMPLE_linear_minmax.yaml \
  --values-json '{"water_level_m": 0.25, "rainfall_mm": 5.0, "temperature_c": 15.0}'
```

---

## Configuration

Each experiment uses a YAML config file (e.g., `configs/EXAMPLE_linear_minmax.yaml`). Key sections:

### Data

```yaml
data:
  target_column: water_level_cm        # Target variable
  feature_columns: [...]               # Input features
  class_names: ["low", "high"]         # For binary/multiclass classification
  test_size: 0.2
  validation_size: 0.1
  timestamp_column: timestamp
  # Temporal split (optional):
  validation_start: "2024-01-01"
  test_start: "2025-01-01"
```

### Feature Engineering

```yaml
feature_engineering:
  generate_lag_features: true
  lag_hours: 72                         # Max lag in hours
  generate_seasonal_features: true
  generate_rolling_aggregates: true
```

### Model

```yaml
model:
  name: linear_classifier              # or: logistic_regression, mlp
  # Model-specific params (e.g., hidden_dims for MLP)
```

### Training

```yaml
training:
  learning_rate: 0.001
  epochs: 50
  batch_size: 32
  weight_decay: 0.0001
```

---

## Common Tasks

### Check Available Experiments

```bash
python -c "from src.experiments.registry import ExperimentFactory; print(ExperimentFactory.list())"
```

### Run Tests

```bash
pytest -q                              # All tests
pytest tests/test_preprocessing.py -v  # Single module
pytest -k "lag"                        # Tests matching keyword
```

### Clean Up Old Outputs

```bash
rm -rf reports/<experiment_name>
rm -f models/<experiment_name>/checkpoint.pt
```

### View Config

```bash
cat configs/EXAMPLE_linear_minmax.yaml
```

---

## Output Structure

After a full pipeline run:

```java
reports/
  <experiment_name>/
    training_history.csv              # Epoch metrics
    training_summary.json             # Final training stats
    training_curves.png               # Visualization
    confusion_matrix.png              # Prediction accuracy breakdown
    predictions.csv                   # Test predictions + confidence
    evaluation.json                   # Row & event-level metrics
    explainability_report.md          # Top features (SHAP)
    feature_importance.csv            # All features ranked
    shap_summary.png                  # SHAP beeswarm
    shap_bar.png                      # SHAP bar chart
data/
  processed/
    train.csv                         # Preprocessed training split
    validation.csv
    test.csv
models/
  <experiment_name>/
    checkpoint.pt                     # Best model weights
    preprocessor.json                 # Feature statistics (scaler state)
```

---

## Interpretation Guide

### Event Metrics (binary classification: "low" vs "high" water)

- **event_true_positives**: High-water episodes detected correctly.
- **event_false_positives**: Non-events predicted as high water.
- **event_recall**: % of true episodes caught (aim: > 0.8 to avoid missing floods).
- **event_precision**: % of predicted episodes that are real (aim: > 0.7 to avoid false alarms).
- **onset_error_hours**: Hours between predicted and actual episode start (negative = early, positive = late).
- **false_alarm_rate**: False alarms per total rows (aim: < 0.05).

### Seasonal Breakdown

Metrics computed separately for each season (winter/spring/summer/autumn) and year to assess model stability and domain-specific performance. High variation across seasons suggests the model captures seasonal differences; low precision in a particular season may indicate underrepresented events.

### Feature Importance (SHAP)

Top features show which inputs drive predictions. High-ranking features include:

- Lags (e.g., `rainfall_mm_lag_6h`, `temperature_c_lag_24h`): water level responds to past conditions.
- Cumulative rainfall (e.g., `rain_24h_sum`): captures watershed saturation.
- Seasonal features (e.g., `month`, `season`): reveals seasonal patterns.

---

## Troubleshooting

### "Config validation failed"

Check that all required keys are in your YAML (see template files in `configs/`).

### "Model checkpoint not found"

Run `train_model` first, or check that the path in config points to an existing `.pt` file.

### "Timestamp column contains invalid values"

Ensure `timestamp_column` in config matches the actual column name and is in ISO format (YYYY-MM-DD HH:MM:SS).

### "None of the lag columns found in DataFrame"

Ensure feature engineering is enabled and the meteorological columns (rainfall_mm, temperature_c, etc.) are present in the raw data.

### Tests Failing

1. Check Python version (≥3.10 recommended).
2. Reinstall dependencies: `pip install -r requirements.txt --upgrade`.
3. Run one test to get detailed error: `pytest tests/test_preprocessing.py::test_handle_missing_values_with_valid_zero_strategy -vv`.

---

## Next Steps

1. **Event Detection**: Implement O1–O4 rule logic in [src/events/detectors/](../src/events/detectors/) (currently placeholders).
2. **Confidence Calibration**: Add probability scaling to improve calibration of high-water risk scores.
3. **PDF Reports**: Generate formatted PDF with figures and narrative (currently markdown + CSV only).
4. **Seasonal Analysis**: Detailed hypothesis tests and cross-validation per season.
5. **User Notebook**: Create interactive Jupyter notebook walkthrough in [notebooks/](../notebooks/).

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md#-prioritized-implementation-backlog) for detailed backlog.

---

## Registered Experiments and Default Configs

The repository registers the following experiments with default configuration files:

- `linear_classifier` — `configs/linear_water_level.yaml`
- `logistic_regression` — `configs/logistic_water_level.yaml`
- `mlp_classifier` — `configs/mlp_water_level.yaml`

Use these default configs to quickly reproduce example runs or as starting templates for new experiments.

## Questions?

Refer to:

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — What's done, partial, and planned.
- [checklist.md](checklist.md) — Project requirements mapped to implementation.
- [experiment_pipeline.md](experiment_pipeline.md) — Walkthrough of CLI commands.
- [project_sheet.md](project_sheet.md) — Original project goals and deliverables.
