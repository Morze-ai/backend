"""Implements missing value imputation and scaling strategies for water level datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import normalize_text_frame

HYDROLOGICAL_YEAR_START_MONTH = 11

DAILY_MEASUREMENT_RENAMES = {
    "PSKDSZS": "station_code",
    "PSNZWP": "station_name",
    "KDNRZK": "watercourse_name",
    "COROKH": "hydrological_year",
    "COMSCH": "hydrological_month",
    "CODZIEN": "day_of_month",
    "COSTAN": "water_level_cm",
    "COPRZP": "flow_m3_s",
    "COPTMP": "water_temperature_c",
    "COMSCK": "calendar_month_source",
}

ICE_COVERAGE_RENAMES = {
    "PSKDSZS": "station_code",
    "PSNZWP": "station_name",
    "KDNRZK": "watercourse_name",
    "ZJROKH": "hydrological_year",
    "ZJMSCH": "hydrological_month",
    "ZJDZIEN": "day_of_month",
    "ZJGRLD": "ice_thickness_cm",
    "ZJKODZJ": "ice_code",
    "ZJKDPRC": "ice_coverage_tenths",
    "ZJZRST": "vegetation_code",
}


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
    "valid-zero": ImputationStrategy(
        name="valid-zero",
        treat_zero_as_missing=False,  # Preserve zeros in Vistula data
        small_gap_threshold=6,
        large_gap_strategy="seasonal",
        years_to_search=2,
        interpolation_window=3,
    ),
    "invalid-zero": ImputationStrategy(
        name="invalid-zero",
        treat_zero_as_missing=True,  # Treat zeros as missing in Port data
        small_gap_threshold=6,
        large_gap_strategy="seasonal",
        years_to_search=2,
        interpolation_window=3,
    ),
}


def handle_missing_values(
    df: pd.DataFrame, strategy: ImputationStrategy | str = "invalid-zero"
) -> pd.DataFrame:
    """
    Handle missing values with adaptive strategy based on dataset characteristics.

    Args:
        df: DataFrame with 'timestamp' and 'water_level_m' columns
        strategy: ImputationStrategy object or strategy name ("valid-zero" or "invalid-zero")

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


def normalize_station_names(
    df: pd.DataFrame, columns: tuple[str, ...] = ("station_name", "watercourse_name")
) -> pd.DataFrame:
    """Repair station and location name columns."""

    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(
                lambda value: value if pd.isna(value) else str(value)
            )
    normalized = normalize_text_frame(normalized)
    return normalized


def reorder_columns_like_reference(df: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    """Place reference columns first while keeping any extras at the end."""

    ordered_columns = [column for column in reference_columns if column in df.columns]
    ordered_columns.extend([column for column in df.columns if column not in ordered_columns])
    return df.loc[:, ordered_columns]


def hydrological_month_to_calendar_month(
    hydrological_month: pd.Series, start_month: int = HYDROLOGICAL_YEAR_START_MONTH
) -> pd.Series:
    """Translate hydrological month numbers into calendar month numbers."""

    month_values = pd.to_numeric(hydrological_month, errors="coerce")
    return (((start_month - 1 + month_values - 1) % 12) + 1).astype("Int64")


def add_calendar_date_from_hydrological_columns(
    df: pd.DataFrame,
    *,
    year_column: str,
    month_column: str,
    day_column: str,
    start_month: int = HYDROLOGICAL_YEAR_START_MONTH,
) -> pd.DataFrame:
    """Derive calendar date columns from hydrological year, month, and day fields."""

    normalized = df.copy()
    normalized[year_column] = pd.to_numeric(normalized[year_column], errors="coerce").astype(
        "Int64"
    )
    normalized[month_column] = pd.to_numeric(normalized[month_column], errors="coerce").astype(
        "Int64"
    )
    normalized[day_column] = pd.to_numeric(normalized[day_column], errors="coerce").astype("Int64")

    calendar_month = hydrological_month_to_calendar_month(
        normalized[month_column], start_month=start_month
    )
    calendar_year = normalized[year_column] + (calendar_month < start_month).astype("Int64")

    normalized["calendar_year"] = calendar_year.astype("Int64")
    normalized["calendar_month"] = calendar_month
    normalized["calendar_day"] = normalized[day_column]
    timestamp_components = pd.DataFrame(
        {
            "year": normalized["calendar_year"].astype("Float64"),
            "month": normalized["calendar_month"].astype("Float64"),
            "day": normalized["calendar_day"].astype("Float64"),
        }
    )
    normalized["timestamp"] = pd.to_datetime(
        timestamp_components,
        errors="coerce",
    )
    return normalized


def clean_daily_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily water measurements into a canonical processed schema."""

    cleaned = df.rename(columns=DAILY_MEASUREMENT_RENAMES).copy()
    cleaned = normalize_station_names(cleaned)
    cleaned = add_calendar_date_from_hydrological_columns(
        cleaned,
        year_column="hydrological_year",
        month_column="hydrological_month",
        day_column="day_of_month",
    )
    return reorder_columns_like_reference(
        cleaned,
        [
            "timestamp",
            "station_code",
            "station_name",
            "watercourse_name",
            "calendar_year",
            "calendar_month",
            "calendar_day",
            "hydrological_year",
            "hydrological_month",
            "day_of_month",
            "water_level_cm",
            "flow_m3_s",
            "water_temperature_c",
            "calendar_month_source",
        ],
    )


def clean_ice_and_vegetation_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ice and vegetation coverage measurements into a canonical schema."""

    cleaned = df.rename(columns=ICE_COVERAGE_RENAMES).copy()
    cleaned = normalize_station_names(cleaned)
    cleaned = add_calendar_date_from_hydrological_columns(
        cleaned,
        year_column="hydrological_year",
        month_column="hydrological_month",
        day_column="day_of_month",
    )
    return reorder_columns_like_reference(
        cleaned,
        [
            "timestamp",
            "station_code",
            "station_name",
            "watercourse_name",
            "calendar_year",
            "calendar_month",
            "calendar_day",
            "hydrological_year",
            "hydrological_month",
            "day_of_month",
            "ice_thickness_cm",
            "ice_code",
            "ice_coverage_tenths",
            "vegetation_code",
        ],
    )


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


@dataclass(frozen=True)
class SplitResult:
    """Train/validation/test split container."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


PreprocessorStats = dict[str, dict[str, float | str]]


def _build_scaler(strategy: str) -> Scaler:
    strategy_key = strategy.strip().lower()
    if strategy_key == "zscore":
        return ZScoreScaler()
    if strategy_key == "minmax":
        return MinMaxScaler()
    if strategy_key == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unknown preprocessing strategy: '{strategy}'. Expected zscore|minmax|robust."
    )


def fit_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    strategy: str,
) -> PreprocessorStats:
    """Fit per-feature scaling parameters on the given frame."""

    if not feature_columns:
        raise ValueError("At least one feature column is required for preprocessing.")

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns missing in frame: {missing}")

    stats: PreprocessorStats = {}
    for column in feature_columns:
        scaler = _build_scaler(strategy)
        fitted = scaler.fit(frame, column)
        if isinstance(fitted, ZScoreScaler):
            stats[column] = {
                "strategy": "zscore",
                "mean": float(fitted.mean if fitted.mean is not None else 0.0),
                "std": float(fitted.std if fitted.std is not None else 1.0),
            }
        elif isinstance(fitted, MinMaxScaler):
            min_value = float(fitted.min if fitted.min is not None else 0.0)
            max_value = float(fitted.max if fitted.max is not None else min_value)
            stats[column] = {
                "strategy": "minmax",
                "min": min_value,
                "max": max_value,
            }
        elif isinstance(fitted, RobustScaler):
            stats[column] = {
                "strategy": "robust",
                "median": float(fitted.median if fitted.median is not None else 0.0),
                "iqr": float(fitted.iqr if fitted.iqr is not None else 1.0),
            }
        else:
            raise ValueError(f"Unsupported scaler implementation for column '{column}'.")

    return stats


def apply_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stats: PreprocessorStats,
) -> pd.DataFrame:
    """Apply fitted scaling parameters to a frame."""

    transformed = frame.copy()
    missing = [column for column in feature_columns if column not in transformed.columns]
    if missing:
        raise ValueError(f"Feature columns missing in frame: {missing}")

    for column in feature_columns:
        if column not in stats:
            raise ValueError(f"Missing preprocessing statistics for feature column '{column}'.")
        raw_stats = stats[column]
        strategy = str(raw_stats.get("strategy", "")).lower()

        values = pd.to_numeric(transformed[column], errors="coerce")
        if strategy == "zscore":
            mean = float(raw_stats.get("mean", 0.0))
            std = float(raw_stats.get("std", 1.0))
            transformed[column] = (values - mean) / (std if std != 0 else 1.0)
        elif strategy == "minmax":
            min_value = float(raw_stats.get("min", 0.0))
            max_value = float(raw_stats.get("max", min_value))
            denominator = max_value - min_value
            transformed[column] = (values - min_value) / (denominator if denominator != 0 else 1.0)
        elif strategy == "robust":
            median = float(raw_stats.get("median", 0.0))
            iqr = float(raw_stats.get("iqr", 1.0))
            transformed[column] = (values - median) / (iqr if iqr != 0 else 1.0)
        else:
            raise ValueError(f"Unknown preprocessing strategy for column '{column}': '{strategy}'.")

    return transformed


def _stratify_target_or_none(frame: pd.DataFrame, target_column: str) -> pd.Series | None:
    """Return a stratification target only when class counts make stratified split valid."""

    target = frame[target_column]
    counts = target.value_counts(dropna=False)
    if len(counts) < 2:
        return None
    if int(counts.min()) < 2:
        return None
    return target


def split_dataset(
    frame: pd.DataFrame,
    target_column: str,
    test_size: float,
    validation_size: float,
    random_seed: int,
) -> SplitResult:
    """Split a dataset into train, validation, and test partitions."""

    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing in frame.")

    if test_size + validation_size >= 1.0:
        raise ValueError("The sum of test_size and validation_size must be less than 1.0.")

    first_stratify = _stratify_target_or_none(frame, target_column)
    split_one = train_test_split(
        frame,
        test_size=test_size,
        stratify=first_stratify,
        random_state=random_seed,
    )
    train_validation = cast(pd.DataFrame, split_one[0])
    test = cast(pd.DataFrame, split_one[1])
    validation_fraction = validation_size / (1.0 - test_size)
    second_stratify = _stratify_target_or_none(train_validation, target_column)
    split_two = train_test_split(
        train_validation,
        test_size=validation_fraction,
        stratify=second_stratify,
        random_state=random_seed,
    )
    train = cast(pd.DataFrame, split_two[0])
    validation = cast(pd.DataFrame, split_two[1])
    return SplitResult(train=train, validation=validation, test=test)
