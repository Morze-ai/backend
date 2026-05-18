# SeaData Project Usage Guide

This guide explains how to use the SeaData water level anomaly detection and explainability pipeline.

## 🚀 Quick Start

To run the full pipeline (training, evaluation, and reporting for all models):

```bash
make run
```

This will:
1. Fetch and preprocess all datasets.
2. Train Linear, Logistic, and MLP models.
3. Generate evaluation metrics and comparison reports.
4. Create SHAP explainability reports and premium PDF summaries.

## 🛠 Command Line Interface (CLI)

### 1. Training & Evaluation
Run a specific experiment:
```bash
uv run python -m src.cli.run_experiment configs/mlp_water_level.yaml
```

### 2. Explainability (SHAP & PDF)
Generate a visual report for a specific model:
```bash
make explain CONFIG=configs/mlp_water_level.yaml
```
Output: `reports/mlp_water_level/explainability_report.pdf`

### 3. Visualizations
Generate exploratory plots (histograms and pairplots):
```bash
uv run python -m src.cli.visualize configs/mlp_water_level.yaml --include-exploratory
```
Output: `reports/global/mlp_water_level_feature_histograms.png` and `pairplot.png`.

## 🔄 Continuous Evaluation & Forecasting

The continuous evaluation pipeline fetches real-time meteorological forecasts (from IMGW, Open-Meteo, and Stormglass APIs) and runs inference across $+1d$, $+3d$, and $+7d$ horizons. It applies Platt scaling calibration, SHAP explainability, and rule-based event detectors (with a robust thaw detector that prevents false positives outside winter/spring months).

### 1. Running the Pipeline
You can trigger the pipeline with a single zero-parameter command. It automatically reads the default configuration (usually `configs/mlp_water_level.yaml`) from your `.env` file under `CONTINUOUS_DEFAULT_CONFIG`:

```bash
make continuous-predict
```

This command executes the pipeline, prints the evaluation JSON directly to stdout, and saves the output files:
- **Raw JSON Output**: Saved to `reports/<config_name>/continuous_latest.json`
- **Markdown Report**: Saved to `reports/<config_name>/continuous_forecast_<date>.md`
- **Forecast Plot**: Saved to `reports/<config_name>/continuous_forecast_<date>.png`

### 2. Rich 3-Panel Forecast Plots
The generated `.png` plot is a multi-panel visualization summarizing the forecast details:
- **Top Panel**: Risk level score vs. Model confidence score for each horizon ($+1d$, $+3d$, $+7d$).
- **Middle Panel**: Forecasted weather conditions, showing Rainfall (blue bars, left axis) and Temperature (red line, right axis).
- **Bottom Panel**: Barometric pressure (green line) overlaid with active system warnings (e.g., missing APIs, invalid bounds).

These plots are automatically updated on every pipeline run and stored in the `reports/<config_name>/` directory.

### 3. Exposing via FastAPI REST API
To make the forecasts available to frontend dashboards, start the FastAPI API server:

```bash
make api-server
```

The server will be available at `http://localhost:8000`. You can inspect the interactive OpenAPI docs at `http://localhost:8000/docs`.

#### API Endpoints:
- `GET /continuous/forecast`: Returns the aggregated dashboard view, detailing current risk, confidence, expected onset window, and $+1d/+3d/+7d$ horizon summaries.
- `GET /continuous/status`: Returns system health check, detailing external API status (IMGW, Open-Meteo, Stormglass), last evaluation time, and warnings count.
- `GET /continuous/predictions/{horizon}`: Fetches detailed forecast entries for a specific horizon (`+1d`, `+3d`, or `+7d`).
- `GET /continuous/latest`: Retrieves the full raw JSON evaluation payload.
- `POST /continuous/refresh`: Manually triggers an on-demand refresh run of the pipeline.

## 🧠 Key Features

### Probability Calibration
We use **Platt Scaling** to ensure that predicted probabilities are well-calibrated. A confidence score of 90% means the event has a ~90% historical probability of occurrence.

### Factor Attribution (O1-O4)
The pipeline automatically identifies the primary driver for detected events:
- **O1: Flash Flood**: Driven by short-term intense rainfall.
- **O2: Long Rainfall**: Driven by saturated catchment and persistent rain.
- **O3: Thaw**: Driven by rising temperatures and snowmelt.
- **O4: Seasonal**: Driven by long-term seasonal patterns.

### Historical Confidence
Calculated by analyzing how often rule-based detections successfully matched historical high-water events in the training set.

## 📊 Outputs

- **PDF Reports**: Premium executive summaries with feature importance and factor analysis.
- **CSV Metrics**: Detailed row-level and event-level performance metrics.
- **Visuals**: Distribution plots, training curves, and confusion matrices.

---
*For technical details, see [src/README.md](src/README.md) or the [checklist.md](docs/checklist.md).*
