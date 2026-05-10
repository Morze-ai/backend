"""Feature engineering for water level prediction."""

from __future__ import annotations

import pandas as pd


def calculate_rain_sums(
    df: pd.DataFrame,
    column: str = "rainfall_mm",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Calculates rolling sums for rainfall over specified windows (in hours).
    """
    if windows is None:
        windows = [1, 3, 6, 12, 24]

    df = df.copy()
    for window in windows:
        df[f"rain_{window}h_sum"] = df[column].rolling(window=window, min_periods=1).sum()

    return df


def calculate_temp_delta(
    df: pd.DataFrame,
    column: str = "temperature_c",
    window: int = 24,
) -> pd.DataFrame:
    """
    Calculates the temperature change over the last X hours.
    """
    df = df.copy()
    # temp_delta_24h = current_temp - temp_24h_ago
    df[f"temp_delta_{window}h"] = df[column].diff(periods=window)

    return df


def calculate_thaw_flag(
    df: pd.DataFrame,
    column: str = "temperature_c",
    window: int = 24,
) -> pd.DataFrame:
    """
    Calculates a flag indicating if the temperature crossed 0°C from below in the last X hours.
    thaw_flag (czy temperatura przekroczyła 0°C w ciągu ostatnich 24h)
    """
    df = df.copy()

    # A thaw occurs if the current temperature is > 0 AND
    # there was a sub-zero temperature in the lookback window.
    has_subzero = (df[column] < 0).rolling(window=window, min_periods=1).max().astype(bool)
    is_above_zero = df[column] > 0

    df["thaw_flag"] = (is_above_zero & has_subzero).astype(int)

    return df


def calculate_soil_saturation(
    df: pd.DataFrame,
    rainfall_column: str = "rainfall_mm",
    alpha: float = 0.9,
    window: int = 24,
) -> pd.DataFrame:
    """
    Calculates a soil saturation indicator based on antecedent precipitation.
    Uses an Exponentially Weighted Moving Average (EWMA) as a proxy for saturation.
    """
    df = df.copy()

    # Antecedent Precipitation Index (API) proxy
    # We use EWMA which is a common way to model soil moisture depletion
    df["soil_saturation_index"] = df[rainfall_column].ewm(alpha=1 - alpha, adjust=False).mean()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all engineered features to the dataframe.
    """
    df = calculate_rain_sums(df)
    df = calculate_temp_delta(df)
    df = calculate_thaw_flag(df)
    df = calculate_soil_saturation(df)

    return df
