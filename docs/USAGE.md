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
