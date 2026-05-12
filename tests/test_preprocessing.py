"""Verifies dataset splitting and feature scaling helpers for standard, min-max, and robust preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    IMPUTATION_STRATEGIES,
    MinMaxScaler,
    RobustScaler,
    ZScoreScaler,
    add_calendar_features,
    add_cyclical_encodings,
    add_seasonal_features,
    handle_missing_values,
    split_dataset,
)


def create_sample_dataframe(rows: int = 100) -> pd.DataFrame:
    """Helper to create a sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(rows) * 10 + 50,
            "feature_2": np.random.randn(rows) * 5 + 25,
            "target": np.random.choice(["low", "medium", "high"], rows),
        }
    )


def create_temporal_dataframe() -> pd.DataFrame:
    """Helper to create a timestamped dataframe for temporal split tests."""
    timestamps = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    values = np.arange(len(timestamps), dtype=float)
    targets = np.where(values % 2 == 0, "low", "high")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature_1": values,
            "target": targets,
        }
    )


def test_zscore_scaler_fit_and_apply() -> None:
    """Test ZScoreScaler fit and apply operations."""
    df = create_sample_dataframe()
    scaler = ZScoreScaler()

    # Fit the scaler
    scaler.fit(df, "feature_1")
    assert scaler.mean is not None
    assert scaler.std is not None

    # Apply the scaler
    scaled_df = scaler.apply(df, "feature_1")
    scaled_col = scaled_df["feature_1"]

    # Check that scaled values have mean ≈ 0 and std ≈ 1
    assert abs(scaled_col.mean()) < 1e-10  # Near zero
    assert abs(scaled_col.std() - 1.0) < 1e-10  # Near one


def test_zscore_scaler_without_fit_raises_error() -> None:
    """Test that applying ZScoreScaler without fit raises error."""
    df = create_sample_dataframe()
    scaler = ZScoreScaler()

    with pytest.raises(ValueError, match="must be fitted"):
        scaler.apply(df, "feature_1")


def test_minmax_scaler_fit_and_apply() -> None:
    """Test MinMaxScaler fit and apply operations."""
    df = create_sample_dataframe()
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(df, "feature_1")
    assert scaler.min is not None
    assert scaler.max is not None

    # Apply the scaler
    scaled_df = scaler.apply(df, "feature_1")
    scaled_col = scaled_df["feature_1"]

    # Check that scaled values are between 0 and 1
    assert scaled_col.min() >= 0.0
    assert scaled_col.max() <= 1.0


def test_minmax_scaler_without_fit_raises_error() -> None:
    """Test that applying MinMaxScaler without fit raises error."""
    df = create_sample_dataframe()
    scaler = MinMaxScaler()

    with pytest.raises(ValueError, match="must be fitted"):
        scaler.apply(df, "feature_1")


def test_robust_scaler_fit_and_apply() -> None:
    """Test RobustScaler fit and apply operations."""
    df = create_sample_dataframe()
    scaler = RobustScaler()

    # Fit the scaler
    scaler.fit(df, "feature_1")
    assert scaler.median is not None
    assert scaler.iqr is not None

    # Apply the scaler
    scaled_df = scaler.apply(df, "feature_1")
    # RobustScaler should not raise errors
    assert not scaled_df.empty
    assert "feature_1" in scaled_df.columns


def test_robust_scaler_without_fit_raises_error() -> None:
    """Test that applying RobustScaler without fit raises error."""
    df = create_sample_dataframe()
    scaler = RobustScaler()

    with pytest.raises(ValueError, match="must be fitted"):
        scaler.apply(df, "feature_1")


def test_split_dataset_basic() -> None:
    """Test basic dataset splitting functionality."""
    df = create_sample_dataframe(rows=100)
    result = split_dataset(
        frame=df,
        target_column="target",
        test_size=0.2,
        validation_size=0.1,
        random_seed=42,
    )

    # Verify splits exist and have correct size
    assert len(result.train) > 0
    assert len(result.validation) > 0
    assert len(result.test) > 0

    # Verify proportions (with some tolerance)
    total = len(result.train) + len(result.validation) + len(result.test)
    assert total == len(df)

    expected_test = int(len(df) * 0.2)
    expected_validation = int(len(df) * 0.1 * (1 - 0.2))
    assert len(result.test) == expected_test
    # Allow ±3 tolerance for rounding in stratified split
    assert abs(len(result.validation) - expected_validation) <= 3


def test_split_dataset_stratification() -> None:
    """Test that split maintains class distribution."""
    df = create_sample_dataframe(rows=300)

    # Get class distribution in original data
    original_dist = df["target"].value_counts(normalize=True)

    result = split_dataset(
        frame=df,
        target_column="target",
        test_size=0.2,
        validation_size=0.1,
        random_seed=42,
    )

    # Check that train set has similar distribution
    train_dist = result.train["target"].value_counts(normalize=True)
    for class_name in original_dist.index:
        if class_name in train_dist.index:
            # Allow ±5% tolerance for stratification
            assert abs(original_dist[class_name] - train_dist[class_name]) < 0.05


def test_split_dataset_missing_target_column() -> None:
    """Test that split raises error when target column is missing."""
    df = create_sample_dataframe()

    with pytest.raises(ValueError, match="Target column"):
        split_dataset(
            frame=df,
            target_column="nonexistent_column",
            test_size=0.2,
            validation_size=0.1,
            random_seed=42,
        )


def test_split_dataset_invalid_sizes() -> None:
    """Test that split raises error when sizes are invalid."""
    df = create_sample_dataframe()

    with pytest.raises(ValueError, match="sum of test_size and validation_size"):
        split_dataset(
            frame=df,
            target_column="target",
            test_size=0.6,
            validation_size=0.5,  # Sum = 1.1, which is > 1.0
            random_seed=42,
        )


def test_split_dataset_temporal_holdout() -> None:
    """Test that temporal splitting respects chronological boundaries."""
    df = create_temporal_dataframe()

    result = split_dataset(
        frame=df,
        target_column="target",
        test_size=0.2,
        validation_size=0.1,
        random_seed=42,
        split_strategy="temporal",
        timestamp_column="timestamp",
        validation_start="2023-10-01",
        test_start="2024-01-01",
    )

    assert result.train["timestamp"].max() < pd.Timestamp("2023-10-01")
    assert result.validation["timestamp"].min() >= pd.Timestamp("2023-10-01")
    assert result.validation["timestamp"].max() < pd.Timestamp("2024-01-01")
    assert result.test["timestamp"].min() >= pd.Timestamp("2024-01-01")
    assert len(result.train) + len(result.validation) + len(result.test) == len(df)


def test_handle_missing_values_with_valid_zero_strategy() -> None:
    """Test handle_missing_values with valid-zero strategy."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=10, freq="D"),
            "water_level_m": [1.0, np.nan, np.nan, 1.5, 2.0, np.nan, 2.5, 3.0, 3.5, 0.0],
        }
    )

    result = handle_missing_values(df, strategy="valid-zero")
    # Zeros should NOT be treated as missing in valid-zero strategy
    assert result["water_level_m"].iloc[-1] == 0.0


def test_handle_missing_values_with_invalid_zero_strategy() -> None:
    """Test handle_missing_values with invalid-zero strategy."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=10, freq="D"),
            "water_level_m": [1.0, np.nan, np.nan, 1.5, 2.0, np.nan, 2.5, 3.0, 3.5, 0.0],
        }
    )

    result = handle_missing_values(df, strategy="invalid-zero")
    # Zeros should be treated as missing and imputed in invalid-zero strategy
    assert result["water_level_m"].notna().all()


def test_handle_missing_values_unknown_strategy() -> None:
    """Test that unknown strategy raises error."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=10, freq="D"),
            "water_level_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="Unknown strategy"):
        handle_missing_values(df, strategy="unknown_strategy")


def test_imputation_strategies_are_defined() -> None:
    """Test that predefined imputation strategies exist."""
    assert len(IMPUTATION_STRATEGIES) > 0
    assert "valid-zero" in IMPUTATION_STRATEGIES
    assert "invalid-zero" in IMPUTATION_STRATEGIES


def test_add_calendar_features() -> None:
    """Test that calendar features are correctly extracted from timestamp."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=10, freq="D"),
            "value": range(10),
        }
    )

    result = add_calendar_features(df)

    # Check that new columns were added
    assert "month" in result.columns
    assert "day_of_week" in result.columns
    assert "day_of_year" in result.columns
    assert "week_of_year" in result.columns

    # Check specific values for first row (2021-01-01 is a Friday)
    assert result["month"].iloc[0] == 1  # January
    assert result["day_of_week"].iloc[0] == 4  # Friday (0=Monday, 4=Friday)
    assert result["day_of_year"].iloc[0] == 1  # First day of year


def test_add_calendar_features_missing_column() -> None:
    """Test that error is raised when timestamp column is missing."""
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="Timestamp column"):
        add_calendar_features(df)


def test_add_cyclical_encodings() -> None:
    """Test that cyclical encodings (sin/cos) are correctly generated."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=12, freq="ME"),
            "month": range(1, 13),
        }
    )

    result = add_cyclical_encodings(df, features=["month"], periods={"month": 12})

    # Check that sin/cos columns were added
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns

    # Check that values are bounded between -1 and 1
    assert result["month_sin"].min() >= -1.0
    assert result["month_sin"].max() <= 1.0
    assert result["month_cos"].min() >= -1.0
    assert result["month_cos"].max() <= 1.0

    # Check specific values: sin(2π * 3/12) should be 1.0 (month 3 = March, 90 degrees)
    march_idx = 2  # 0-indexed
    assert abs(result["month_sin"].iloc[march_idx] - 1.0) < 0.001


def test_add_cyclical_encodings_missing_feature() -> None:
    """Test that error is raised when feature column is missing."""
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match=r"Feature.*not found"):
        add_cyclical_encodings(df, features=["month"], periods={"month": 12})


def test_add_cyclical_encodings_missing_period() -> None:
    """Test that error is raised when period is not specified."""
    df = pd.DataFrame({"month": [1, 2, 3]})

    with pytest.raises(ValueError, match="Period not specified"):
        add_cyclical_encodings(df, features=["month"], periods={})


def test_add_seasonal_features() -> None:
    """Test that seasonal features are correctly assigned."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-01-15",  # Winter
                    "2021-04-15",  # Spring
                    "2021-07-15",  # Summer
                    "2021-10-15",  # Autumn
                    "2021-12-15",  # Winter
                ]
            ),
            "value": range(5),
        }
    )

    result = add_seasonal_features(df)

    # Check that season columns were added
    assert "season" in result.columns
    assert "season_winter" in result.columns
    assert "season_spring" in result.columns
    assert "season_summer" in result.columns
    assert "season_autumn" in result.columns

    # Check specific season assignments
    assert result["season"].iloc[0] == "winter"
    assert result["season"].iloc[1] == "spring"
    assert result["season"].iloc[2] == "summer"
    assert result["season"].iloc[3] == "autumn"
    assert result["season"].iloc[4] == "winter"

    # Check one-hot encodings
    assert result["season_winter"].iloc[0] == 1
    assert result["season_spring"].iloc[0] == 0
    assert result["season_spring"].iloc[1] == 1
    assert result["season_summer"].iloc[2] == 1


def test_add_seasonal_features_missing_column() -> None:
    """Test that error is raised when timestamp column is missing."""
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="Timestamp column"):
        add_seasonal_features(df)
