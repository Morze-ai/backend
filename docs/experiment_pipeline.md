# Experiment Pipeline Implementation

## Walkthrough

### 1. Fetch or normalize raw data

```bash
python -m src.cli.fetch_data configs/EXAMPLE_linear_minmax.yaml --source-csv data/raw/dead-vistula-river-water-level-2021-2025.csv
```

### 2. Preprocess and split

```bash
python -m src.cli.preprocess_data configs/EXAMPLE_linear_minmax.yaml
```

### 3. Train

```bash
python -m src.cli.train_model configs/EXAMPLE_linear_minmax.yaml
```

### 4. Evaluate

```bash
python -m src.cli.evaluate_model configs/EXAMPLE_linear_minmax.yaml
```

### 5. Visualize

```bash
python -m src.cli.visualize configs/EXAMPLE_linear_minmax.yaml
```

### 6. Predict one sample

```bash
python -m src.cli.predict configs/EXAMPLE_linear_minmax.yaml --values-json '{"water_level_m": 0.25}'
```

### 7. Run all stages together

```bash
python -m src.cli.run_experiment configs/EXAMPLE_linear_minmax.yaml
```

### 8. Summarize reports

```bash
python -m src.cli.report_summary --reports-root reports --output-csv reports/global/experiment_summary.csv
```

### 9. Compare experiments

```bash
python -m src.cli.compare_experiments configs/EXAMPLE_compare_experiments.yaml
```

## Notes

- Logistic regression mode is binary-only and requires two class labels.
- For multiclass tasks, use `linear_classifier` or `mlp_classifier`.
