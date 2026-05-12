
# Lag Features & Time Series Engineering

This document explains how lag features are integrated into the water level prediction pipeline.

## Overview

**Lag features** are historical values of a variable from previous time steps. They're essential for time series prediction because they capture temporal dependencies and delayed effects.

**Example:**
```
timestamp     rainfall_mm    rainfall_mm_lag_1h    rainfall_mm_lag_6h    rainfall_mm_lag_24h
2021-01-02        0                 -                    -                     -
2021-01-02        1               0                      -                     -
2021-01-02        2               1                    0                       -
2021-01-02        3               2                    1                       -
2021-01-03        1               3                    2                       0
2021-01-03        0               1                    3                       1
```

This allows the model to learn patterns like: "If it rained 24 hours ago (heavy rain), and now it's raining again, the water level will rise."

## Architecture

### Files Involved

| File | Purpose |
|------|---------|
| [src/data/feature_engineering.py](src/data/feature_engineering.py) | Core lag/rolling/seasonal feature generation |
| [src/experiments/base.py](src/experiments/base.py) | Integration point - automatically detects & generates lags |
| [tests/test_feature_engineering.py](tests/test_feature_engineering.py) | Comprehensive test suite (19 tests) |
| [configs/EXAMPLE_with_lag_features.yaml](configs/EXAMPLE_with_lag_features.yaml) | Example config demonstrating lag features |

### Processing Pipeline

```
Raw Data (CSV)
      ↓
[1] Data Loading (timestamp, water_level_m, rainfall_mm, temperature_c, ...)
      ↓
[2] Missing Value Imputation
      ↓
[3] LAG FEATURE GENERATION ← THIS IS NEW
      ├─ Detect available weather columns
      ├─ Generate lags (1h, 2h, ..., 72h)
      ├─ Generate rolling aggregates (mean, max, min, std)
      ├─ Generate seasonal features (month, day, season, etc.)
      └─ Drop warmup rows (first N=max_lags rows)
      ↓
[4] Train/Validation/Test Split
      ↓
[5] Normalization (MinMax/StandardScaler)
      ↓
[6] Model Training (Linear/Logistic/MLP)
      ↓
[7] Evaluation & Predictions
```

## Functions

### 1. `generate_lag_features()`

Creates shifted historical copies of features.

```python
from src.data.feature_engineering import generate_lag_features
import pandas as pd

df = pd.DataFrame({
    "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
    "rainfall_mm": [0, 1, 2, 3, ...],
    "temperature_c": [5, 6, 7, 8, ...],
})

# Generate 72 lags for specified columns
df_with_lags = generate_lag_features(
    df,
    timestamp_column="timestamp",
    lag_columns={"rainfall_mm": 72, "temperature_c": 72},
)

# Result columns: rainfall_mm_lag_1h, rainfall_mm_lag_2h, ..., rainfall_mm_lag_72h
# Plus original rainfall_mm column
```

**Output:**
- New columns: `{column}_lag_{N}h` for N=1..max_lags
- NaN values for rows without sufficient history
- Shape: same as input (rows) × (original_cols + num_lags*len(lag_columns))

### 2. `generate_rolling_features()`

Creates aggregates over time windows (mean, max, min, std).

```python
from src.data.feature_engineering import generate_rolling_features

df_with_rolling = generate_rolling_features(
    df,
    window_hours=[3, 6, 12, 24],
    agg_functions=["mean", "max", "min", "std"],
    columns_to_aggregate=["rainfall_mm", "temperature_c"],
)

# Result: rainfall_mm_mean_3h, rainfall_mm_max_3h, ..., temperature_c_std_24h
```

**Use case:** "What was the average temperature over the last 6 hours?"

### 3. `generate_seasonal_features()`

Extracts temporal patterns from timestamps.

```python
from src.data.feature_engineering import generate_seasonal_features

df_with_seasonal = generate_seasonal_features(df, timestamp_column="timestamp")

# New columns:
# - month: 1-12
# - day_of_year: 1-366
# - day_of_week: 0-6 (Monday=0)
# - hour_of_day: 0-23
# - is_weekend: 0/1
# - season: "winter", "spring", "summer", "autumn"
# - is_growing_season: 0/1 (April-October)
```

**Use case:** Model learns that spring flooding is different from winter patterns.

### 4. `drop_initial_lag_rows()`

Removes rows without complete lag history.

```python
from src.data.feature_engineering import drop_initial_lag_rows

# After generating 72-hour lags, first 72 rows have NaN values
df_clean = drop_initial_lag_rows(df_with_lags, max_lag_hours=72)

# Result: DataFrame with first 72 rows removed, index reset
```

## Integration in Experiments

The `BaseExperiment.preprocess()` method ([src/experiments/base.py](src/experiments/base.py)) automatically integrates lag features:

```python
def preprocess(self, frame: pd.DataFrame) -> None:
    """Automatically generate lag features if weather data exists."""
    
    processed_frame = frame.copy()
    
    # Auto-detect weather columns
    weather_columns = {
        "rainfall_mm": 72,
        "temperature_c": 72,
        "pressure_hpa": 72
    }
    available_weather = {
        col: lags for col, lags in weather_columns.items() 
        if col in frame.columns
    }
    
    # Generate lags if weather data exists
    if available_weather:
        processed_frame = generate_lag_features(
            processed_frame,
            timestamp_column="timestamp",
            lag_columns=available_weather,
        )
        
        # Drop warmup period (72 rows for 72-hour lags)
        max_lags = max(available_weather.values())
        processed_frame = drop_initial_lag_rows(
            processed_frame,
            max_lag_hours=max_lags,
        )
        
        # Add new lag feature names to feature_columns
        for col in available_weather:
            for lag_hour in range(1, available_weather[col] + 1):
                lag_col_name = f"{col}_lag_{lag_hour}h"
                if lag_col_name not in self.config.data.feature_columns:
                    self.config.data.feature_columns.append(lag_col_name)
    
    # Continue with split, normalization, etc.
    # ...
```

**Key points:**
- ✅ Automatic detection of available weather columns
- ✅ Generates lags only if weather data exists
- ✅ Seamless integration - config doesn't need to specify lag columns manually
- ✅ Handles missing weather gracefully (skips those columns)

## Configuration

### Minimal Example (No Lag Features)

```yaml
data:
  target_column: timestamp
  feature_columns:
    - water_level_m
  test_size: 0.2
  validation_size: 0.2
```

The model will only use `water_level_m` as a feature. No historical context.

### With Lag Features (Recommended)

```yaml
data:
  target_column: timestamp
  feature_columns:
    - water_level_m
    # Lag features will be AUTOMATICALLY added if weather data exists!
    # You don't need to list them manually.
  test_size: 0.2
  validation_size: 0.2
```

If your data has `rainfall_mm`, `temperature_c`, `pressure_hpa` columns:
- The system automatically generates 72 lags for each
- Total features: 1 + (3 × 72) = 217 features

### Why Not Manual?

❌ Don't manually list all lag columns:
```yaml
feature_columns:
  - water_level_m
  - rainfall_mm_lag_1h
  - rainfall_mm_lag_2h
  # ... 70 more lines for each lag ...
  - temperature_c_lag_72h
```

✅ Let the system handle it:
```yaml
feature_columns:
  - water_level_m
  # Lags auto-generated during preprocess()
```

## Example Workflow

### Step 1: Check Your Data

```bash
head data/raw/dead-vistula-river-water-level-2021-2025.csv
# Columns: timestamp, water_level_m

head data/raw/hail-mountain-weather-data-2021-2025.csv
# Columns: timestamp, rainfall_mm, temperature_c, humidity_percentage, pressure_hpa
```

### Step 2: Run with Lag Features

```bash
# Use the provided example config
python -m src.cli.run_experiment configs/EXAMPLE_with_lag_features.yaml
```

### Step 3: Check Preprocessing Output

```bash
head data/processed/vistula_mlp_with_lag_features/train.csv
# Will have many columns: water_level_m, water_level_m_lag_1h, ..., 
# rainfall_mm, rainfall_mm_lag_1h, ..., temperature_c, temperature_c_lag_1h, ...
```

### Step 4: Monitor Training

```bash
cat reports/vistula_mlp_with_lag_features/training_history.csv
# Accuracy should improve thanks to temporal context
```

## Performance Impact

### Data Size
- Original: ~43,824 hourly samples (2021-2025)
- After lag generation: 43,824 samples (same rows)
- After dropping warmup: 43,752 samples (72 rows removed)
- Features: 1 → ~217 (with 3 weather columns × 72 lags)

### Training Time
- Linear model: ~1 second
- MLP (128-64-32): ~15-30 seconds (depends on epochs/batch size)

### Memory Usage
- 1 GB+ with 216 lag features × 43K samples
- Consider reducing lags or feature selection if constrained

## Advanced Usage

### Custom Lag Ranges

```python
# Use fewer lags for faster training
lag_columns = {
    "rainfall_mm": 24,      # Only 1 day of history
    "temperature_c": 24,
    "pressure_hpa": 24,
}
```

### Rolling Features Too

```python
# In base.py, you could add:
df_with_rolling = generate_rolling_features(
    df_with_lags,
    window_hours=[3, 6, 12, 24],
    agg_functions=["mean", "max", "min"],
    columns_to_aggregate=["rainfall_mm", "temperature_c"],
)
```

### Seasonal Features Too

```python
# In base.py:
df_with_seasonal = generate_seasonal_features(df_with_rolling)
# Adds: month, day_of_year, season, is_growing_season, etc.
```

## Tests

All lag feature functions are tested:

```bash
uv run pytest tests/test_feature_engineering.py -v
```

**Coverage:**
- ✅ Lag creation & shifting
- ✅ Initial NaN handling
- ✅ Rolling aggregates
- ✅ Seasonal features
- ✅ Warmup period dropping
- 19 tests, all passing

## Troubleshooting

### "AttributeError: 'DataFrame' has no attribute 'rainfall_mm'"

**Cause:** Your data doesn't have weather columns (rainfall_mm, temperature_c, pressure_hpa).

**Solution:** The system will gracefully skip lag generation for missing columns. Check your CSV:
```bash
head -1 data/raw/your_data.csv
```

### "ValueError: max_lag_hours >= DataFrame length"

**Cause:** You have fewer samples than the lag window (e.g., 50 samples but 72 lags).

**Solution:** Reduce lags or use more data:
```python
lag_columns = {"rainfall_mm": 12}  # Use 12h instead of 72h
```

### No Improvement After Adding Lags

**Possible reasons:**
1. Weather data isn't predictive for this location
2. Lags need different range (try 24h, 48h, 12h)
3. Model needs more epochs to learn temporal patterns
4. Try MLP instead of linear (handles nonlinear temporal patterns better)

## Next Steps

See [data_cleaning_strategy.md](data_cleaning_strategy.md) for missing value handling details.

See [experiment_pipeline.md](experiment_pipeline.md) for full workflow examples.

For event detection with lag features, see [events/README.md](../src/events/README.md) (future).
