# Data Cleaning Strategy - Missing Value Imputation

## Summary

- **Dead Vistula River**: 1,483 missing values
- **Northern Port**: 2,336 missing values

## Problem Analysis

### Missing Values Identified

- Missing values marked as `"-"` in CSV files
- Distribution:
  - Small gaps: 1-83 consecutive missing records (scattered throughout 2021-2022)
  - Large gaps:
    - the disaster of June 2023 -> 42 missing days (1,008 records)

### Special Considerations

- **Dead Vistula River** has many valid zero values -> all zeros preserved

- **Northern Port** also has no valid zeros -> all zeros imputed (I assume, it's not like we know. Maybe the sea disappeared for a moment, who freaking knows anymore?)

## Imputation Strategy

### 1. **Small Gaps (≤ 6 measurements)**

**Method**: Nearest Neighbor Mean (KNN)

- Takes mean of 3 measurements before + 3 measurements after the gap
- Result rounded to 2 decimal places
- Preserves local temporal patterns

**Rationale**: These gaps are typically 6 hours or less, preserving continuity with surrounding data points

**Example**:

```yaml
Before gap:  [-0.28, -0.25, -0.27]
After gap:   [-0.29, -0.28, -0.28]
Imputed:     mean([-0.28, -0.25, -0.27, -0.29, -0.28, -0.28]) = -0.27
```

### 2. **Large Gaps (> 6 measurements)**

**Method**: Seasonal Averaging

- For each missing timestamp, find corresponding date-hour in previous/next 2 years
- Average available historical values for that date-hour
- Falls back to linear interpolation if no seasonal match found (for timestamps near the start in 2021)

**Rationale**:

- Captures seasonal patterns (e.g., spring flooding, winter low levels)
- Respects multi-year climate variations
- Appropriate for gaps spanning weeks/months

### 3. **Fallback Strategy**

When seasonal averaging finds no matching historical data:

- Uses linear interpolation between nearest valid neighbors (up to 24 hours away)
- If only one neighbor available, repeats that value
- Ensures 100% coverage without gaps

## Implementation Details

### File: `src/data/preprocessing.py`

**Key Functions:**

- `handle_missing_values()` - Main entry point
- `_identify_missing_groups()` - Detects consecutive gaps
- `_interpolate_small_gap()` - KNN imputation
- `_impute_seasonal_average()` - Seasonal averaging with fallback

**Parameters:**

```python
handle_missing_values(
    df: pd.DataFrame,
    dataset_name: str = "port",  # or "vistula"
    small_gap_threshold: int = 6,  # Imputation strategy threshold
    large_gap_strategy: str = "seasonal"
)
```

## Processed Files

Output files saved to `data/processed/`:

- `dead-vistula-cleaned.csv`
- `northern-port-cleaned.csv`
