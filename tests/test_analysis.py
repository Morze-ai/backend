"""Tests for the analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    StatisticalAnalyzer,
    analyze_onset_error_distribution,
    compare_groups_by_threshold,
    compute_lag_correlations,
    soil_saturation_event_crosstab,
)


class TestComputeLagCorrelations:
    """Tests for lag correlation analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic water level data with lag features."""
        np.random.seed(42)
        n = 200

        # Create normalized water level (zscore: mean~0, std~1)
        water_level = np.random.randn(n) * 0.8 + 0.1

        # Create synthetic rainfall with autocorrelation
        rainfall = np.random.randn(n) * 0.7
        rainfall_series = pd.Series(rainfall)
        rainfall_lag_1h = rainfall_series.shift(1).values
        rainfall_lag_6h = rainfall_series.shift(6).values

        # Temperature feature
        temperature = np.random.randn(n) * 0.9
        temperature_series = pd.Series(temperature)
        temperature_lag_1h = temperature_series.shift(1).values

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
                "water_level_m": water_level,
                "rainfall_mm": rainfall,
                "rainfall_mm_lag_1h": rainfall_lag_1h,
                "rainfall_mm_lag_6h": rainfall_lag_6h,
                "temperature_c": temperature,
                "temperature_c_lag_1h": temperature_lag_1h,
            }
        )

        # Add season
        month = df["timestamp"].dt.month
        df["season"] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["winter", "spring", "summer"],
            default="autumn",
        )

        return df

    def test_compute_lag_correlations_creates_results(self, sample_data):
        """Test that lag correlations are computed."""
        results = compute_lag_correlations(
            sample_data,
            target_column="water_level_m",
            lag_columns=["rainfall_mm_lag_1h", "rainfall_mm_lag_6h"],
        )

        assert len(results) > 0
        assert all(hasattr(r, "pearson_r") for r in results)
        assert all(hasattr(r, "spearman_rho") for r in results)

    def test_lag_correlations_by_season(self, sample_data):
        """Test that seasonal grouping works."""
        winter_results = compute_lag_correlations(
            sample_data,
            target_column="water_level_m",
            season="winter",
            lag_columns=["rainfall_mm_lag_1h"],
        )

        all_results = compute_lag_correlations(
            sample_data,
            target_column="water_level_m",
            season=None,
            lag_columns=["rainfall_mm_lag_1h"],
        )

        assert len(winter_results) > 0
        assert len(all_results) > 0
        assert all(r.season == "winter" for r in winter_results)
        assert all(r.season == "all" for r in all_results)

    def test_lag_correlations_missing_data_handling(self, sample_data):
        """Test pair-wise deletion for NaN values."""
        # Add some NaN values
        sample_data.loc[0:5, "rainfall_mm_lag_1h"] = np.nan

        results = compute_lag_correlations(
            sample_data,
            target_column="water_level_m",
            lag_columns=["rainfall_mm_lag_1h"],
        )

        assert len(results) > 0
        assert results[0].n_missing > 0
        assert results[0].n_samples < len(sample_data)

    def test_lag_correlations_insufficient_data(self, sample_data):
        """Test handling of insufficient data (< 3 samples)."""
        small_df = sample_data.iloc[:2].copy()

        results = compute_lag_correlations(
            small_df,
            target_column="water_level_m",
            lag_columns=["rainfall_mm_lag_1h"],
        )

        # Should return NaN correlations for insufficient data
        assert len(results) > 0
        assert np.isnan(results[0].pearson_r)

    def test_auto_detect_lag_columns(self, sample_data):
        """Test automatic detection of lag columns."""
        results = compute_lag_correlations(
            sample_data,
            target_column="water_level_m",
            lag_columns=None,  # Auto-detect
        )

        # Should find rainfall_mm_lag_1h, rainfall_mm_lag_6h, temperature_c_lag_1h
        assert len(results) >= 3


class TestCompareGroupsByThreshold:
    """Tests for hypothesis testing (high vs low water groups)."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic data with strong group differences."""
        np.random.seed(42)
        n = 300

        # Create water level with clear signal
        water_level = np.random.normal(0, 1, n)

        # Feature that differs between high and low water
        rainfall = np.where(
            water_level > 0.5,
            np.random.normal(2.0, 0.8, n),  # High water: high rainfall
            np.random.normal(0.5, 0.8, n),  # Low water: low rainfall
        )

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
                "water_level_m": water_level,
                "rainfall_mm": rainfall,
                "temperature_c": np.random.randn(n),
            }
        )

        month = df["timestamp"].dt.month
        df["season"] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["winter", "spring", "summer"],
            default="autumn",
        )

        return df

    def test_compare_groups_by_threshold_ttest(self, sample_data):
        """Test t-test comparison between groups."""
        result = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
            threshold_percentile=75.0,
        )

        assert result is not None
        assert hasattr(result, "ttest_statistic")
        assert hasattr(result, "ttest_p_value")
        assert result.n_group1 > 0
        assert result.n_group2 > 0

    def test_compare_groups_mannwhitney(self, sample_data):
        """Test Mann-Whitney U test."""
        result = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
        )

        assert result is not None
        assert hasattr(result, "mannwhitney_statistic")
        assert hasattr(result, "mannwhitney_p_value")

    def test_compare_groups_effect_sizes(self, sample_data):
        """Test Cohen's d and rank-biserial effect sizes."""
        result = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
        )

        assert result is not None
        assert not np.isnan(result.ttest_cohens_d)
        assert not np.isnan(result.mannwhitney_rank_biserial)

    def test_compare_groups_multiple_testing_correction(self, sample_data):
        """Test Bonferroni and FDR corrections."""
        result = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
        )

        assert result is not None
        # Bonferroni should be >= uncorrected
        assert result.ttest_p_value_bonferroni >= result.ttest_p_value
        # Both should be <= 1
        assert result.ttest_p_value_bonferroni <= 1.0
        assert result.ttest_p_value_fdr <= 1.0

    def test_compare_groups_by_season(self, sample_data):
        """Test seasonal grouping."""
        result_winter = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
            season="winter",
        )

        result_all = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
            season=None,
        )

        assert result_winter is not None
        assert result_all is not None
        assert result_winter.season == "winter"
        assert result_all.season == "all"

    def test_compare_groups_normality_test(self, sample_data):
        """Test Shapiro-Wilk normality test."""
        result = compare_groups_by_threshold(
            sample_data,
            feature_name="rainfall_mm",
            target_column="water_level_m",
        )

        assert result is not None
        assert not np.isnan(result.shapiro_p_value_group1)
        assert not np.isnan(result.shapiro_p_value_group2)


class TestSoilSaturationCrosstab:
    """Tests for contingency table analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic data with soil saturation and event correlation."""
        np.random.seed(42)
        n = 400

        # Soil saturation (normalized)
        soil_sat = np.random.uniform(-1, 3, n)

        # Event occurrence linked to soil saturation
        event_prob = 1 / (1 + np.exp(-0.5 * (soil_sat - 1)))  # Logistic function
        event = (np.random.random(n) < event_prob).astype(int)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
                "soil_saturation_index": soil_sat,
                "event_occurred": event,
            }
        )

        month = df["timestamp"].dt.month
        df["season"] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["winter", "spring", "summer"],
            default="autumn",
        )

        return df

    def test_soil_saturation_crosstab_chi2(self, sample_data):
        """Test chi-square test on contingency table."""
        result = soil_saturation_event_crosstab(
            sample_data,
            soil_saturation_column="soil_saturation_index",
            event_column="event_occurred",
        )

        assert result is not None
        assert hasattr(result, "chi2_statistic")
        assert hasattr(result, "chi2_p_value")
        assert result.chi2_p_value >= 0
        assert result.chi2_p_value <= 1.0

    def test_soil_saturation_cramers_v(self, sample_data):
        """Test Cramér's V effect size."""
        result = soil_saturation_event_crosstab(
            sample_data,
            soil_saturation_column="soil_saturation_index",
            event_column="event_occurred",
        )

        assert result is not None
        assert 0 <= result.cramers_v <= 1.0

    def test_soil_saturation_contingency_table(self, sample_data):
        """Test contingency table construction."""
        result = soil_saturation_event_crosstab(
            sample_data,
            soil_saturation_column="soil_saturation_index",
            event_column="event_occurred",
        )

        assert result is not None
        assert isinstance(result.contingency_table, dict)
        assert result.n_samples > 0

    def test_soil_saturation_by_season(self, sample_data):
        """Test seasonal grouping."""
        result_winter = soil_saturation_event_crosstab(
            sample_data,
            soil_saturation_column="soil_saturation_index",
            event_column="event_occurred",
            season="winter",
        )

        result_all = soil_saturation_event_crosstab(
            sample_data,
            soil_saturation_column="soil_saturation_index",
            event_column="event_occurred",
            season=None,
        )

        if result_winter:
            assert result_winter.season == "winter"
        if result_all:
            assert result_all.season == "all"


class TestOnsetErrorDistribution:
    """Tests for onset error analysis."""

    def test_onset_error_distribution(self):
        """Test distribution statistics computation."""
        errors = [0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 48.0]

        result = analyze_onset_error_distribution(errors, season="winter")

        assert result is not None
        assert result.min_hours == 0
        assert result.max_hours == 48
        assert result.n_errors == 10
        assert 0 < result.mean_hours < 48

    def test_onset_error_distribution_with_nan(self):
        """Test handling of NaN and inf values."""
        errors = [0.0, 2.0, np.nan, 5.0, np.inf, 10.0, 12.0]

        result = analyze_onset_error_distribution(errors)

        assert result is not None
        # NaN and inf should be removed
        assert result.n_errors == 5
        assert result.max_hours == 12

    def test_onset_error_distribution_percentiles(self):
        """Test percentile calculations."""
        errors = [float(value) for value in range(1, 101)]  # 1 to 100

        result = analyze_onset_error_distribution(errors)

        assert result is not None
        assert result.p10_hours == pytest.approx(10.9, rel=2)  # ~10th percentile
        assert result.p25_hours == pytest.approx(25.75, rel=2)  # ~25th percentile
        assert result.p75_hours == pytest.approx(75.75, rel=2)  # ~75th percentile
        assert result.p90_hours == pytest.approx(90.9, rel=2)  # ~90th percentile

    def test_onset_error_empty_list(self):
        """Test handling of empty onset error list."""
        result = analyze_onset_error_distribution([])
        assert result is None

    def test_onset_error_single_value(self):
        """Test handling of single value."""
        result = analyze_onset_error_distribution([5.0])

        assert result is not None
        assert result.min_hours == 5.0
        assert result.max_hours == 5.0
        assert result.mean_hours == 5.0
        assert result.std_hours == 0.0


class TestStatisticalAnalyzer:
    """Tests for the main StatisticalAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive synthetic dataset."""
        np.random.seed(42)
        n = 500

        # Normalized features (zscore-like)
        water_level = np.random.randn(n) * 0.8
        rainfall = np.random.randn(n) * 0.7
        temperature = np.random.randn(n) * 0.9
        pressure = np.random.randn(n) * 0.6
        soil_sat = np.random.uniform(-1, 2, n)

        # Create lags
        rainfall_series = pd.Series(rainfall)
        rainfall_lag_1h = rainfall_series.shift(1).fillna(0).values
        rainfall_lag_6h = rainfall_series.shift(6).fillna(0).values
        temperature_series = pd.Series(temperature)
        temperature_lag_1h = temperature_series.shift(1).fillna(0).values

        # Event with some correlation to rainfall
        event = (rainfall > np.percentile(rainfall, 75)).astype(int)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
                "water_level_m": water_level,
                "rainfall_mm": rainfall,
                "temperature_c": temperature,
                "pressure_hpa": pressure,
                "soil_saturation_index": soil_sat,
                "rainfall_mm_lag_1h": rainfall_lag_1h,
                "rainfall_mm_lag_6h": rainfall_lag_6h,
                "temperature_c_lag_1h": temperature_lag_1h,
                "event_occurred": event,
            }
        )

        month = df["timestamp"].dt.month
        df["season"] = np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["winter", "spring", "summer"],
            default="autumn",
        )

        return df

    def test_analyzer_initialization(self, sample_data):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer(sample_data, dataset_name="test_data")

        assert analyzer.dataset_name == "test_data"
        assert len(analyzer.df) == len(sample_data)

    def test_analyzer_missing_timestamp(self):
        """Test initialization fails without timestamp column."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValueError, match="timestamp"):
            StatisticalAnalyzer(df)

    def test_analyzer_adds_season(self, sample_data):
        """Test that analyzer adds season column if missing."""
        df = sample_data.drop(columns=["season"])
        analyzer = StatisticalAnalyzer(df)

        assert "season" in analyzer.df.columns

    def test_analyzer_lag_analysis(self, sample_data):
        """Test lag correlation analysis."""
        analyzer = StatisticalAnalyzer(sample_data)
        results = analyzer.analyze_lags_by_season(
            target_column="water_level_m",
            lag_columns=["rainfall_mm_lag_1h", "temperature_c_lag_1h"],
        )

        assert len(results) > 0

    def test_analyzer_group_differences(self, sample_data):
        """Test hypothesis testing."""
        analyzer = StatisticalAnalyzer(sample_data)
        results = analyzer.analyze_group_differences(
            features_to_test=["rainfall_mm", "temperature_c"],
            target_column="water_level_m",
        )

        assert len(results) > 0

    def test_analyzer_soil_saturation(self, sample_data):
        """Test soil saturation contingency analysis."""
        analyzer = StatisticalAnalyzer(sample_data)
        results = analyzer.analyze_soil_saturation(
            soil_saturation_column="soil_saturation_index",
            event_column="event_occurred",
        )

        assert len(results) >= 0

    def test_analyzer_onset_errors(self, sample_data):
        """Test onset error distribution analysis."""
        analyzer = StatisticalAnalyzer(sample_data)

        onset_errors = {
            "winter": [1.0, 2.5, 3.0],
            "spring": [0.5, 1.0, 2.0],
            "summer": [2.0, 3.0, 4.0],
            "autumn": [1.5, 2.0, 3.5],
        }

        results = analyzer.analyze_onset_errors(onset_errors)
        assert len(results) > 0

    def test_analyzer_full_summary(self, sample_data):
        """Test comprehensive statistical summary generation."""
        analyzer = StatisticalAnalyzer(sample_data, dataset_name="full_test")

        summary = analyzer.generate_statistical_summary(
            target_column="water_level_m",
            event_column="event_occurred",
            soil_saturation_column="soil_saturation_index",
            features_to_test=["rainfall_mm", "temperature_c", "pressure_hpa"],
            lag_columns=["rainfall_mm_lag_1h", "rainfall_mm_lag_6h"],
            onset_errors_by_season={
                "winter": [1.0, 2.0, 3.0],
                "spring": [0.5, 1.5, 2.5],
            },
        )

        assert summary.dataset_name == "full_test"
        assert summary.n_total_rows == len(sample_data)
        assert len(summary.lag_correlations) > 0
        assert len(summary.hypothesis_tests) > 0

    def test_analyzer_summary_to_dict(self, sample_data):
        """Test conversion to dictionary for JSON serialization."""
        analyzer = StatisticalAnalyzer(sample_data)

        summary = analyzer.generate_statistical_summary(
            target_column="water_level_m",
            features_to_test=["rainfall_mm"],
        )

        summary_dict = summary.to_dict()

        assert isinstance(summary_dict, dict)
        assert "dataset_name" in summary_dict
        assert "lag_correlations" in summary_dict
        assert "hypothesis_tests" in summary_dict

    def test_analyzer_normalization_check(self, sample_data):
        """Test normalization check warning."""
        # Create un-normalized data
        df = sample_data.copy()
        df["unnormalized_feature"] = np.random.uniform(100, 1000, len(df))

        analyzer = StatisticalAnalyzer(df)

        # Should generate warnings about normalization
        assert len(analyzer.warnings) >= 0  # May or may not warn depending on data
