"""Generate advanced feature engineering: lag features for time series prediction."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def generate_lag_features(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    lag_columns: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Generate lag features for specified columns up to a maximum number of hours.

    Args:
        df: DataFrame with timestamp and feature columns to lag.
        timestamp_column: Name of the timestamp column. Must be datetime-like.
        lag_columns: Dict mapping column names to max lag hours.
            Default: {"rainfall_mm": 72, "temperature_c": 72, "pressure_hpa": 72}

    Returns:
        DataFrame with original columns plus lag feature columns.
        New columns named: {original}_lag_{N}h (e.g., rainfall_mm_lag_1h, rainfall_mm_lag_72h)
        Rows with insufficient history get NaN values.

    Example:
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
        ...     "rainfall_mm": np.random.rand(100),
        ...     "temperature_c": np.random.randn(100) * 10 + 5,
        ... })
        >>> df_with_lags = generate_lag_features(df)
        >>> print(df_with_lags.columns)
        # Shows rainfall_mm_lag_1h, rainfall_mm_lag_2h, ..., temperature_c_lag_72h, etc.
    """
    # Set default lag columns if not provided
    if lag_columns is None:
        lag_columns = {
            "rainfall_mm": 72,
            "temperature_c": 72,
            "pressure_hpa": 72,
        }

    result = df.copy()

    # Ensure timestamp is datetime
    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(result[timestamp_column])

    # Validate that lag columns exist in DataFrame
    missing_cols = [col for col in lag_columns if col not in result.columns]
    if missing_cols:
        # Log warning but continue with available columns
        available = {k: v for k, v in lag_columns.items() if k in result.columns}
        if not available:
            raise ValueError(f"None of the lag columns found in DataFrame. Available: {list(result.columns)}")
        lag_columns = available

    # Generate lag features for each specified column
    # Build all lag columns at once to avoid dataframe fragmentation
    lag_data = {}
    for column, max_lags in lag_columns.items():
        for lag_hour in range(1, max_lags + 1):
            lag_col_name = f"{column}_lag_{lag_hour}h"
            lag_data[lag_col_name] = result[column].shift(lag_hour)

    # Add all lag columns at once
    if lag_data:
        lag_df = pd.DataFrame(lag_data, index=result.index)
        result = pd.concat([result, lag_df], axis=1)

    return result


def generate_rolling_features(
    df: pd.DataFrame,
    window_hours: list[int] | None = None,
    agg_functions: list[str] | None = None,
    columns_to_aggregate: list[str] | None = None,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Generate rolling window aggregates (mean, max, min, std) for time series features.

    Args:
        df: DataFrame with timestamp and feature columns.
        window_hours: List of window sizes in hours (default: [3, 6, 12, 24])
        agg_functions: List of aggregation functions to apply (default: ["mean", "max", "min", "std"])
        columns_to_aggregate: List of column names to aggregate (default: ["rainfall_mm", "temperature_c"])
        timestamp_column: Name of the timestamp column.

    Returns:
        DataFrame with original columns plus rolling aggregate columns.
        New columns named: {column}_{agg}_{window}h (e.g., rainfall_mm_mean_3h, temperature_c_max_24h)

    Example:
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
        ...     "rainfall_mm": np.random.rand(100),
        ... })
        >>> df_rolled = generate_rolling_features(df)
        >>> print(df_rolled.filter(like="rainfall_mm_mean").columns)
        # Shows rainfall_mm_mean_3h, rainfall_mm_mean_6h, etc.
    """
    if window_hours is None:
        window_hours = [3, 6, 12, 24]

    if agg_functions is None:
        agg_functions = ["mean", "max", "min", "std"]

    if columns_to_aggregate is None:
        columns_to_aggregate = ["rainfall_mm", "temperature_c"]

    result = df.copy()

    # Ensure timestamp is datetime
    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(result[timestamp_column])
        result = result.set_index(timestamp_column)

    # Validate columns exist
    missing_cols = [col for col in columns_to_aggregate if col not in result.columns]
    if missing_cols:
        available = [col for col in columns_to_aggregate if col in result.columns]
        if not available:
            raise ValueError(
                f"None of the aggregation columns found. Available: {list(result.columns)}"
            )
        columns_to_aggregate = available

    # Generate rolling aggregates
    for window in window_hours:
        for col in columns_to_aggregate:
            for func in agg_functions:
                rolling = result[col].rolling(window=window, min_periods=1)
                agg_result = getattr(rolling, func)()
                new_col_name = f"{col}_{func}_{window}h"
                result[new_col_name] = agg_result

    # Reset index to restore timestamp as column
    if timestamp_column in df.columns:
        result = result.reset_index()

    return result


def generate_seasonal_features(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Generate seasonal and temporal features from timestamp.

    Args:
        df: DataFrame with timestamp column.
        timestamp_column: Name of the timestamp column.

    Returns:
        DataFrame with additional columns:
        - month: 1-12
        - day_of_year: 1-366
        - day_of_week: 0-6 (Monday=0)
        - hour_of_day: 0-23
        - is_weekend: 0 or 1
        - season: "spring", "summer", "autumn", "winter"
        - is_growing_season: 0 or 1 (April-October in Northern hemisphere)
    """
    result = df.copy()

    # Ensure timestamp is datetime
    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(result[timestamp_column])
        ts = result[timestamp_column]
    else:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    # Basic temporal features
    result["month"] = ts.dt.month
    result["day_of_year"] = ts.dt.dayofyear
    result["day_of_week"] = ts.dt.dayofweek
    result["hour_of_day"] = ts.dt.hour
    result["is_weekend"] = ((ts.dt.dayofweek >= 5).astype(int))

    # Hydrological season (November start, per Polish convention)
    # Months: 11, 12 = fall/winter; 1, 2, 3 = winter/spring; 4-10 = spring/summer/fall
    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:  # 9, 10, 11
            return "autumn"

    result["season"] = ts.dt.month.apply(get_season)

    # Growing season (April-October)
    result["is_growing_season"] = ((ts.dt.month >= 4) & (ts.dt.month <= 10)).astype(int)

    return result


def drop_initial_lag_rows(
    df: pd.DataFrame,
    max_lag_hours: int = 72,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Drop rows that don't have complete lag history.

    Args:
        df: DataFrame with lag features.
        max_lag_hours: Maximum lag hours used (drops first N rows where N = max_lag_hours)
        timestamp_column: Name of timestamp column.

    Returns:
        DataFrame with first max_lag_hours rows removed.
    """
    if max_lag_hours <= 0:
        return df.copy()

    if max_lag_hours >= len(df):
        raise ValueError(
            f"max_lag_hours ({max_lag_hours}) is >= DataFrame length ({len(df)}). "
            "Not enough data to drop lag warmup period."
        )

    return df.iloc[max_lag_hours:].reset_index(drop=True)
