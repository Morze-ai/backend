"""Tests for lag, rolling, and seasonal feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    drop_initial_lag_rows,
    generate_lag_features,
    generate_rolling_features,
    generate_seasonal_features,
)


class TestGenerateLagFeatures:
    """Tests for generate_lag_features function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with weather data."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
                "rainfall_mm": np.random.rand(100) * 10,
                "temperature_c": np.random.randn(100) * 5 + 10,
                "pressure_hpa": np.random.rand(100) * 20 + 1000,
            }
        )

    def test_generate_lag_features_creates_columns(self, sample_dataframe):
        """Test that lag features are created with correct column names."""
        result = generate_lag_features(sample_dataframe, lag_columns={"rainfall_mm": 3})

        expected_lag_cols = ["rainfall_mm_lag_1h", "rainfall_mm_lag_2h", "rainfall_mm_lag_3h"]
        for col in expected_lag_cols:
            assert col in result.columns

    def test_generate_lag_features_default_lags(self, sample_dataframe):
        """Test that default lag configuration creates 72 lags for each column."""
        result = generate_lag_features(sample_dataframe)

        for col in ["rainfall_mm", "temperature_c", "pressure_hpa"]:
            for lag in range(1, 73):
                assert f"{col}_lag_{lag}h" in result.columns

    def test_lag_values_are_shifted_correctly(self, sample_dataframe):
        """Test that lag values are correct (shifted by N rows)."""
        result = generate_lag_features(sample_dataframe, lag_columns={"rainfall_mm": 2})

        assert result.loc[2, "rainfall_mm_lag_1h"] == sample_dataframe.loc[1, "rainfall_mm"]
        assert result.loc[2, "rainfall_mm_lag_2h"] == sample_dataframe.loc[0, "rainfall_mm"]

    def test_initial_rows_are_nan(self, sample_dataframe):
        """Test that first N rows have NaN for lag_N features (no history)."""
        result = generate_lag_features(sample_dataframe, lag_columns={"rainfall_mm": 3})

        assert pd.isna(result.loc[0, "rainfall_mm_lag_1h"])
        assert pd.isna(result.loc[0, "rainfall_mm_lag_2h"])
        assert pd.isna(result.loc[0, "rainfall_mm_lag_3h"])

        assert not pd.isna(result.loc[1, "rainfall_mm_lag_1h"])
        assert pd.isna(result.loc[1, "rainfall_mm_lag_2h"])
        assert pd.isna(result.loc[1, "rainfall_mm_lag_3h"])

    def test_missing_column_raises_error(self, sample_dataframe):
        """Test that requesting lags for non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="None of the lag columns found"):
            generate_lag_features(sample_dataframe, lag_columns={"nonexistent_col": 5})

    def test_partial_missing_columns_handled(self, sample_dataframe):
        """Test that function handles mix of existing and missing columns gracefully."""
        result = generate_lag_features(
            sample_dataframe,
            lag_columns={"rainfall_mm": 2, "nonexistent_col": 2},
        )

        assert "rainfall_mm_lag_1h" in result.columns
        assert "nonexistent_col_lag_1h" not in result.columns

    def test_dataframe_length_unchanged(self, sample_dataframe):
        """Test that generated DataFrame has same number of rows as input."""
        result = generate_lag_features(sample_dataframe, lag_columns={"rainfall_mm": 72})

        assert len(result) == len(sample_dataframe)


class TestGenerateRollingFeatures:
    """Tests for generate_rolling_features function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=50, freq="h"),
                "rainfall_mm": np.full(50, 5.0),
                "temperature_c": np.full(50, 10.0),
            }
        )

    def test_rolling_mean_is_correct(self, sample_dataframe):
        """Test that rolling mean is calculated correctly."""
        result = generate_rolling_features(
            sample_dataframe,
            window_hours=[3],
            agg_functions=["mean"],
        )

        col_name = "rainfall_mm_mean_3h"
        assert col_name in result.columns
        assert result.loc[5:, col_name].mean() > 4.9

    def test_rolling_features_created(self, sample_dataframe):
        """Test that expected rolling feature columns are created."""
        result = generate_rolling_features(
            sample_dataframe,
            window_hours=[3, 6],
            agg_functions=["mean", "max"],
            columns_to_aggregate=["rainfall_mm"],
        )

        expected_cols = [
            "rainfall_mm_mean_3h",
            "rainfall_mm_max_3h",
            "rainfall_mm_mean_6h",
            "rainfall_mm_max_6h",
        ]
        for col in expected_cols:
            assert col in result.columns


class TestGenerateSeasonalFeatures:
    """Tests for generate_seasonal_features function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
                "value": np.random.rand(100),
            }
        )

    def test_month_feature_created(self, sample_dataframe):
        """Test that month feature is created correctly."""
        result = generate_seasonal_features(sample_dataframe)

        assert "month" in result.columns
        assert result.loc[0, "month"] == 1

    def test_day_of_year_feature_created(self, sample_dataframe):
        """Test that day_of_year feature is created correctly."""
        result = generate_seasonal_features(sample_dataframe)

        assert "day_of_year" in result.columns
        assert result.loc[0, "day_of_year"] == 1

    def test_season_feature_created(self, sample_dataframe):
        """Test that season feature is created."""
        result = generate_seasonal_features(sample_dataframe)

        assert "season" in result.columns
        assert result.loc[0, "season"] == "winter"

    def test_is_weekend_feature(self, sample_dataframe):
        """Test that is_weekend feature is correct."""
        result = generate_seasonal_features(sample_dataframe)

        assert "is_weekend" in result.columns
        assert result.loc[0, "is_weekend"] == 0

    def test_all_temporal_features_created(self, sample_dataframe):
        """Test that all expected temporal features are created."""
        result = generate_seasonal_features(sample_dataframe)

        expected_features = [
            "month",
            "day_of_year",
            "day_of_week",
            "hour_of_day",
            "is_weekend",
            "season",
            "season_code",
            "is_growing_season",
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "hour_of_day_sin",
            "hour_of_day_cos",
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_cyclical_features_are_bounded(self, sample_dataframe):
        """Test cyclical encodings are in [-1, 1] range."""
        result = generate_seasonal_features(sample_dataframe)

        cyclical_columns = [
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "hour_of_day_sin",
            "hour_of_day_cos",
        ]

        for column in cyclical_columns:
            assert result[column].between(-1.0, 1.0).all()


class TestDropInitialLagRows:
    """Tests for drop_initial_lag_rows function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
                "value": range(100),
            }
        )

    def test_drops_correct_number_of_rows(self, sample_dataframe):
        """Test that correct number of rows are dropped."""
        result = drop_initial_lag_rows(sample_dataframe, max_lag_hours=10)

        assert len(result) == len(sample_dataframe) - 10

    def test_index_is_reset(self, sample_dataframe):
        """Test that index is reset after dropping rows."""
        result = drop_initial_lag_rows(sample_dataframe, max_lag_hours=5)

        assert result.index[0] == 0
        assert result.index[-1] == len(result) - 1

    def test_first_row_after_drop_is_correct(self, sample_dataframe):
        """Test that first row after drop has correct value."""
        result = drop_initial_lag_rows(sample_dataframe, max_lag_hours=10)

        assert result.loc[0, "value"] == 10

    def test_zero_lag_hours_returns_copy(self, sample_dataframe):
        """Test that max_lag_hours=0 returns unchanged copy."""
        result = drop_initial_lag_rows(sample_dataframe, max_lag_hours=0)

        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_exceeds_dataframe_length_raises_error(self, sample_dataframe):
        """Test that exceeding dataframe length raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"max_lag_hours .* is >= DataFrame length",
        ):
            drop_initial_lag_rows(sample_dataframe, max_lag_hours=200)
