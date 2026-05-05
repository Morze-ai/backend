"""Implements missing value imputation and scaling strategies for water level datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ImputationStrategy:
    """Configuration for dataset-specific imputation strategies."""

    name: str
    treat_zero_as_missing: bool = False
    small_gap_threshold: int = 6
    large_gap_strategy: str = "seasonal"
    years_to_search: int = 2
    interpolation_window: int = 3


# Predefined strategies for different datasets
IMPUTATION_STRATEGIES = {
    "vistula": ImputationStrategy(
        name="vistula",
        treat_zero_as_missing=False,  # Preserve zeros in Vistula data
        small_gap_threshold=6,
        large_gap_strategy="seasonal",
        years_to_search=2,
        interpolation_window=3,
    ),
    "port": ImputationStrategy(
        name="port",
        treat_zero_as_missing=True,  # Treat zeros as missing in Port data
        small_gap_threshold=6,
        large_gap_strategy="seasonal",
        years_to_search=2,
        interpolation_window=3,
    ),
}


def handle_missing_values(
    df: pd.DataFrame, strategy: ImputationStrategy | str = "port"
) -> pd.DataFrame:
    """
    Handle missing values with adaptive strategy based on dataset characteristics.

    Args:
        df: DataFrame with 'timestamp' and 'water_level_m' columns
        strategy: ImputationStrategy object or strategy name ("vistula" or "port")

    Returns:
        DataFrame with missing values imputed
    """
    # Resolve strategy
    if isinstance(strategy, str):
        if strategy not in IMPUTATION_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {list(IMPUTATION_STRATEGIES.keys())}"
            )
        strategy = IMPUTATION_STRATEGIES[strategy]

    df = df.copy()

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Replace "-" with NaN
    df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")

    # Treat zeros as missing if specified in strategy
    if strategy.treat_zero_as_missing:
        df.loc[df["water_level_m"] == 0, "water_level_m"] = np.nan

    # Identify missing value ranges
    missing_mask = df["water_level_m"].isna()

    if not missing_mask.any():
        return df

    # Find consecutive missing value groups
    missing_groups = _identify_missing_groups(df, missing_mask)

    for start_idx, end_idx in missing_groups:
        gap_size = end_idx - start_idx + 1

        if gap_size <= strategy.small_gap_threshold:
            # Use nearest neighbor interpolation
            _interpolate_small_gap(df, start_idx, end_idx, window=strategy.interpolation_window)
        else:
            # Use seasonal averaging for large gaps
            if strategy.large_gap_strategy == "seasonal":
                _impute_seasonal_average(
                    df, start_idx, end_idx, years_to_search=strategy.years_to_search
                )

    return df


def _identify_missing_groups(df: pd.DataFrame, missing_mask: pd.Series) -> list[tuple[int, int]]:
    """Identify consecutive groups of missing values."""
    missing_indices = np.where(missing_mask)[0]

    if len(missing_indices) == 0:
        return []

    groups = []
    start_idx = missing_indices[0]
    prev_idx = missing_indices[0]

    for idx in missing_indices[1:]:
        if idx - prev_idx > 1:  # Gap found, save group and start new one
            groups.append((start_idx, prev_idx))
            start_idx = idx
        prev_idx = idx

    groups.append((start_idx, prev_idx))
    return groups


def _interpolate_small_gap(df: pd.DataFrame, start_idx: int, end_idx: int, window: int = 3) -> None:
    """
    Fill small gaps using nearest neighbor average (KNN imputation).
    Takes mean of 'window' values before and after the gap.
    """
    # Get values before the gap
    before_idx = max(0, start_idx - window)
    before_values = _numeric_window_values(df.loc[before_idx : start_idx - 1, "water_level_m"])

    # Get values after the gap
    after_idx = min(len(df) - 1, end_idx + window)
    after_values = _numeric_window_values(df.loc[end_idx + 1 : after_idx, "water_level_m"])

    # Combine and calculate mean
    all_nearby = np.concatenate((before_values, after_values))

    if len(all_nearby) > 0:
        fill_value = round(float(all_nearby.mean()), 2)
        df.loc[start_idx:end_idx, "water_level_m"] = fill_value


def _impute_seasonal_average(
    df: pd.DataFrame, start_idx: int, end_idx: int, years_to_search: int = 2
) -> None:
    """
    Fill large gaps using seasonal averaging: find same dates in other years and average them.

    For each missing timestamp, look back/forward in other years and average available values.
    Falls back to linear interpolation if no seasonal data is available.
    """
    for idx in range(start_idx, end_idx + 1):
        current_ts = cast(pd.Timestamp, df.at[idx, "timestamp"])

        # Find same date in nearby years
        seasonal_values = []

        for year_offset in range(-years_to_search, years_to_search + 1):
            if year_offset == 0:
                continue

            try:
                target_year = current_ts.year + year_offset
                target_ts = current_ts.replace(year=target_year)

                # Find exact timestamp in dataframe
                matching_rows = df[df["timestamp"] == target_ts]
                if not matching_rows.empty:
                    val = matching_rows.iloc[0]["water_level_m"]
                    if pd.notna(val):
                        seasonal_values.append(float(val))
            except ValueError:
                # Year doesn't exist (e.g., trying to reach year 2020 when data starts in 2021)
                continue

        if seasonal_values:
            fill_value = round(float(np.mean(seasonal_values)), 2)
            df.loc[idx, "water_level_m"] = fill_value
        else:
            # Fallback: use linear interpolation between neighbors
            before_val = None
            after_val = None

            # Find last non-NaN value before this index
            for i in range(idx - 1, max(idx - 24, -1), -1):  # Look back up to 24 hours
                if pd.notna(df.loc[i, "water_level_m"]):
                    before_val = float(cast(Any, df.loc[i, "water_level_m"]))
                    break

            # Find first non-NaN value after this index
            for i in range(idx + 1, min(idx + 24, len(df))):  # Look forward up to 24 hours
                if pd.notna(df.loc[i, "water_level_m"]):
                    after_val = float(cast(Any, df.loc[i, "water_level_m"]))
                    break

            if before_val is not None and after_val is not None:
                # Linear interpolation
                fill_value = round((before_val + after_val) / 2.0, 2)
                df.loc[idx, "water_level_m"] = fill_value
            elif before_val is not None:
                df.loc[idx, "water_level_m"] = before_val
            elif after_val is not None:
                df.loc[idx, "water_level_m"] = after_val


def _numeric_window_values(series: pd.Series) -> np.ndarray:
    """Return a typed float array for local window calculations."""
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    return numeric.to_numpy(dtype=float, copy=False)


class Scaler:
    """Base class for scaling strategies with fit/apply pattern."""

    def fit(self, df: pd.DataFrame, column: str) -> Scaler:
        """Fit scaler on training data."""
        # TODO: Implement fit logic in subclasses
        raise NotImplementedError

    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply fitted scaler to data."""
        # TODO: Implement apply logic in subclasses
        raise NotImplementedError


class ZScoreScaler(Scaler):
    """Z-score normalization: (x - mean) / std."""

    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, df: pd.DataFrame, column: str) -> ZScoreScaler:
        """Fit z-score scaler on the specified column."""
        self.mean = df[column].mean()
        self.std = df[column].std()
        return self

    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply fitted z-score scaler."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before applying")
        df = df.copy()
        df[column] = (df[column] - self.mean) / self.std
        return df


class MinMaxScaler(Scaler):
    """Min-max normalization: (x - min) / (max - min)."""

    def __init__(self):
        self.min: float | None = None
        self.max: float | None = None

    def fit(self, df: pd.DataFrame, column: str) -> MinMaxScaler:
        """Fit min-max scaler on the specified column."""
        self.min = df[column].min()
        self.max = df[column].max()
        return self

    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply fitted min-max scaler."""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before applying")
        df = df.copy()
        df[column] = (df[column] - self.min) / (self.max - self.min)
        return df


class RobustScaler(Scaler):
    """Robust scaling: (x - median) / IQR."""

    def __init__(self):
        self.median: float | None = None
        self.iqr: float | None = None

    def fit(self, df: pd.DataFrame, column: str) -> RobustScaler:
        """Fit robust scaler on the specified column."""
        self.median = df[column].median()
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        self.iqr = q3 - q1
        return self

    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply fitted robust scaler."""
        if self.median is None or self.iqr is None:
            raise ValueError("Scaler must be fitted before applying")
        df = df.copy()
        if self.iqr != 0:
            df[column] = (df[column] - self.median) / self.iqr
        return df
