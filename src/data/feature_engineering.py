"""Feature engineering utilities for water level prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rain_sums(
    df: pd.DataFrame,
    column: str = "rainfall_mm",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Calculate rolling rainfall sums over provided windows (in hours)."""
    if windows is None:
        windows = [1, 3, 6, 12, 24]

    result = df.copy()
    for window in windows:
        result[f"rain_{window}h_sum"] = result[column].rolling(window=window, min_periods=1).sum()

    return result


def calculate_temp_delta(
    df: pd.DataFrame,
    column: str = "temperature_c",
    window: int = 24,
) -> pd.DataFrame:
    """Calculate temperature delta relative to value from N hours ago."""
    result = df.copy()
    result[f"temp_delta_{window}h"] = result[column].diff(periods=window)
    return result


def calculate_thaw_flag(
    df: pd.DataFrame,
    column: str = "temperature_c",
    window: int = 24,
) -> pd.DataFrame:
    """Calculate thaw flag: current temperature above 0 and sub-zero seen in lookback."""
    result = df.copy()

    has_subzero = (result[column] < 0).rolling(window=window, min_periods=1).max().astype(bool)
    is_above_zero = result[column] > 0

    result["thaw_flag"] = (is_above_zero & has_subzero).astype(int)
    return result


def calculate_soil_saturation(
    df: pd.DataFrame,
    rainfall_column: str = "rainfall_mm",
    alpha: float = 0.9,
    window: int = 24,
) -> pd.DataFrame:
    """Calculate soil saturation proxy with EWMA of rainfall."""
    del window  # kept for API compatibility
    result = df.copy()
    result["soil_saturation_index"] = (
        result[rainfall_column]
        .ewm(
            alpha=1 - alpha,
            adjust=False,
        )
        .mean()
    )
    return result


def calculate_temp_mean(
    df: pd.DataFrame,
    column: str = "temperature_c",
    window: int = 24,
) -> pd.DataFrame:
    """Calculate rolling mean temperature over the provided window."""
    result = df.copy()
    result["temp_mean"] = result[column].rolling(window=window, min_periods=1).mean()
    return result


def calculate_wind_features(
    df: pd.DataFrame,
    speed_column: str = "wind_speed_ms",
    direction_column: str = "wind_direction_deg",
) -> pd.DataFrame:
    """Create wind speed/direction aliases and Cartesian components (U, V)."""
    result = df.copy()

    if speed_column in result.columns:
        result["wind_speed"] = result[speed_column]

    if direction_column in result.columns:
        result["wind_direction"] = result[direction_column]

    if "wind_speed" in result.columns and "wind_direction" in result.columns:
        radians = np.radians(result["wind_direction"])
        result["wind_u"] = result["wind_speed"] * np.sin(radians)
        result["wind_v"] = result["wind_speed"] * np.cos(radians)

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-specific engineered weather features."""
    result = calculate_rain_sums(df)
    result = calculate_temp_delta(result)
    result = calculate_temp_mean(result)
    result = calculate_thaw_flag(result)
    result = calculate_soil_saturation(result)
    result = calculate_wind_features(result)
    return result


def generate_lag_features(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    lag_columns: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Generate lag features for selected columns up to configured lag hours."""
    if lag_columns is None:
        lag_columns = {
            "rainfall_mm": 72,
            "temperature_c": 72,
            "pressure_hpa": 72,
        }

    result = df.copy()

    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(result[timestamp_column])

    missing_cols = [col for col in lag_columns if col not in result.columns]
    if missing_cols:
        available = {k: v for k, v in lag_columns.items() if k in result.columns}
        if not available:
            raise ValueError(
                f"None of the lag columns found in DataFrame. Available: {list(result.columns)}"
            )
        lag_columns = available

    lag_data: dict[str, pd.Series] = {}
    for column, max_lags in lag_columns.items():
        for lag_hour in range(1, max_lags + 1):
            lag_col_name = f"{column}_lag_{lag_hour}h"
            lag_data[lag_col_name] = result[column].shift(lag_hour)

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
    """Generate rolling aggregates (mean/max/min/std) for selected columns."""
    if window_hours is None:
        window_hours = [3, 6, 12, 24]

    if agg_functions is None:
        agg_functions = ["mean", "max", "min", "std"]

    if columns_to_aggregate is None:
        columns_to_aggregate = ["rainfall_mm", "temperature_c"]

    result = df.copy()

    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(result[timestamp_column])
        result = result.set_index(timestamp_column)

    missing_cols = [col for col in columns_to_aggregate if col not in result.columns]
    if missing_cols:
        available = [col for col in columns_to_aggregate if col in result.columns]
        if not available:
            raise ValueError(
                f"None of the aggregation columns found. Available: {list(result.columns)}"
            )
        columns_to_aggregate = available

    for window in window_hours:
        for col in columns_to_aggregate:
            for func in agg_functions:
                rolling = result[col].rolling(window=window, min_periods=1)
                result[f"{col}_{func}_{window}h"] = getattr(rolling, func)()

    if timestamp_column in df.columns:
        result = result.reset_index()

    return result


def generate_seasonal_features(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Generate timestamp-derived seasonal and temporal features."""
    result = df.copy()

    if timestamp_column in result.columns:
        result[timestamp_column] = pd.to_datetime(result[timestamp_column])
        ts = result[timestamp_column]
    else:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    result["month"] = ts.dt.month
    result["day_of_year"] = ts.dt.dayofyear
    result["day_of_week"] = ts.dt.dayofweek
    result["hour_of_day"] = ts.dt.hour
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    def get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return "winter"
        if month in [3, 4, 5]:
            return "spring"
        if month in [6, 7, 8]:
            return "summer"
        return "autumn"

    result["season"] = ts.dt.month.apply(get_season)
    result["is_growing_season"] = ((ts.dt.month >= 4) & (ts.dt.month <= 10)).astype(int)

    return result


def drop_initial_lag_rows(
    df: pd.DataFrame,
    max_lag_hours: int = 72,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Drop leading rows that cannot have complete lag history."""
    del timestamp_column  # kept for API compatibility
    if max_lag_hours <= 0:
        return df.copy()

    if max_lag_hours >= len(df):
        raise ValueError(
            f"max_lag_hours ({max_lag_hours}) is >= DataFrame length ({len(df)}). "
            "Not enough data to drop lag warmup period."
        )

    return df.iloc[max_lag_hours:].reset_index(drop=True)
