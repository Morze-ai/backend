"""Statistical analysis of water level data, lag correlations, and event factors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.schemas import (
    CrosstabResult,
    HypothesisTestResult,
    OnsetErrorDistribution,
    SeasonalLagCorrelation,
    StatisticalSummary,
)


def _bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Apply Bonferroni correction: p_corrected = min(p * m, 1) where m is the number of tests."""
    m = len(p_values)
    if m == 0:
        return []
    return [min(p * m, 1.0) for p in p_values]


def _fdr_correction(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []

    m = len(p_values)
    indexed_p_values = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    sorted_p = [p for _index, p in indexed_p_values]

    # Compute critical values: i/m * alpha
    critical_values = [(rank / m) * 0.05 for rank in range(1, m + 1)]

    # Find largest i where P(i) <= i/m * alpha
    valid_indices = [
        index for index, p_value in enumerate(sorted_p) if p_value <= critical_values[index]
    ]

    if len(valid_indices) == 0:
        return [1.0] * m

    threshold = critical_values[valid_indices[-1]]
    return [min(float(p), threshold) if float(p) <= threshold else 1.0 for p in p_values]


def compute_lag_correlations(
    df: pd.DataFrame,
    target_column: str = "water_level_m",
    season: str | None = None,
    lag_columns: list[str] | None = None,
) -> list[SeasonalLagCorrelation]:
    """
    Compute correlations between lag features and target within a season.

    Args:
        df: DataFrame with timestamp, target_column, and lag feature columns
        target_column: Name of target variable (e.g., 'water_level_m')
        season: Season name (e.g., 'winter', 'spring'). If None, computes across all data.
        lag_columns: List of lag column names to analyze. If None, auto-detect lag_*h columns.

    Returns:
        List of SeasonalLagCorrelation objects.

    Notes:
        - Assumes data is already normalized (zscore/minmax/robust scaling).
        - Missing values (NaN) are handled with pair-wise deletion.
        - Returns both Pearson (parametric) and Spearman (non-parametric) correlations.
    """
    if lag_columns is None:
        # Auto-detect all lag_*h columns
        lag_columns = [col for col in df.columns if "_lag_" in col and col.endswith("h")]

    results: list[SeasonalLagCorrelation] = []

    # Filter by season if specified
    data = df.copy()
    if season is not None:
        if "season" not in data.columns:
            raise ValueError(
                "DataFrame must have 'season' column when season parameter is specified"
            )
        data = data[data["season"] == season]

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    if len(data) == 0:
        return results

    for lag_col in lag_columns:
        if lag_col not in data.columns:
            continue

        # Pair-wise deletion: drop rows with NaN in either column
        clean = data[[target_column, lag_col]].dropna()
        n_missing = len(data) - len(clean)

        if len(clean) < 3:  # Need at least 3 samples for correlation
            results.append(
                SeasonalLagCorrelation(
                    season=season or "all",
                    lag_hours=int(lag_col.split("_lag_")[1].rstrip("h")),
                    feature_name=lag_col,
                    target_column=target_column,
                    pearson_r=np.nan,
                    pearson_p_value=np.nan,
                    spearman_rho=np.nan,
                    spearman_p_value=np.nan,
                    n_samples=len(clean),
                    n_missing=n_missing,
                )
            )
            continue

        # Pearson correlation
        pearson_result = cast(
            tuple[float, float], stats.pearsonr(clean[target_column], clean[lag_col])
        )

        # Spearman correlation (robust to outliers)
        spearman_result = cast(
            tuple[float, float], stats.spearmanr(clean[target_column], clean[lag_col])
        )

        pearson_r = pearson_result[0]
        pearson_p = pearson_result[1]
        spearman_rho = spearman_result[0]
        spearman_p = spearman_result[1]

        results.append(
            SeasonalLagCorrelation(
                season=season or "all",
                lag_hours=int(lag_col.split("_lag_")[1].rstrip("h")),
                feature_name=lag_col,
                target_column=target_column,
                pearson_r=float(pearson_r),
                pearson_p_value=float(pearson_p),
                spearman_rho=float(spearman_rho),
                spearman_p_value=float(spearman_p),
                n_samples=len(clean),
                n_missing=n_missing,
            )
        )

    return results


def compare_groups_by_threshold(
    df: pd.DataFrame,
    feature_name: str,
    target_column: str = "water_level_m",
    threshold_percentile: float = 95.0,
    season: str | None = None,
) -> HypothesisTestResult | None:
    """
    Compare feature values between high and low water level groups using t-test and Mann-Whitney U.

    Args:
        df: DataFrame with feature_name, target_column, and optionally 'season'
        feature_name: Name of feature to compare
        target_column: Name of water level column
        threshold_percentile: Percentile cutoff (default 95%) to define "high water"
        season: Season name. If None, computes across all data.

    Returns:
        HypothesisTestResult with both parametric and non-parametric test results, or None if insufficient data.

    Notes:
        - Uses pair-wise deletion for missing values.
        - Reports both uncorrected and Bonferroni/FDR-corrected p-values.
        - Includes Shapiro-Wilk normality test for each group.
        - Assumes data is already normalized before calling this function.
    """
    if feature_name not in df.columns:
        return None
    if target_column not in df.columns:
        return None

    # Filter by season
    data = df.copy()
    if season is not None:
        if "season" not in data.columns:
            raise ValueError(
                "DataFrame must have 'season' column when season parameter is specified"
            )
        data = data[data["season"] == season]

    if len(data) == 0:
        return None

    # Pair-wise deletion
    clean = data[[feature_name, target_column]].dropna()

    if len(clean) < 4:  # Need at least 4 samples for meaningful tests
        return None

    # Compute threshold
    threshold = clean[target_column].quantile(threshold_percentile / 100.0)

    # Split into groups
    group_high = clean[clean[target_column] >= threshold][feature_name]
    group_low = clean[clean[target_column] < threshold][feature_name]

    if len(group_high) < 2 or len(group_low) < 2:
        return None

    # Drop NaN within groups (extra safety)
    group_high = group_high.dropna()
    group_low = group_low.dropna()

    if len(group_high) < 2 or len(group_low) < 2:
        return None

    # --- T-Test (Parametric) ---
    t_result = cast(tuple[float, float], stats.ttest_ind(group_high, group_low, equal_var=False))
    t_stat = t_result[0]
    t_pval = t_result[1]

    # Cohen's d effect size
    n1, n2 = len(group_high), len(group_low)
    var1, var2 = group_high.var(), group_low.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (group_high.mean() - group_low.mean()) / pooled_std if pooled_std > 0 else 0.0

    # --- Mann-Whitney U (Non-parametric) ---
    u_result = cast(
        tuple[float, float],
        stats.mannwhitneyu(group_high, group_low, alternative="two-sided"),
    )
    u_stat = u_result[0]
    u_pval = u_result[1]

    # Rank-biserial correlation (effect size for Mann-Whitney)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    # --- Normality Tests (Shapiro-Wilk) ---
    shapiro_p1 = (
        cast(tuple[float, float], stats.shapiro(group_high))[1] if len(group_high) >= 3 else np.nan
    )
    shapiro_p2 = (
        cast(tuple[float, float], stats.shapiro(group_low))[1] if len(group_low) >= 3 else np.nan
    )

    # --- Multiple Testing Corrections ---
    # Bonferroni for 2 tests (t-test and Mann-Whitney)
    t_pval_bonf, u_pval_bonf = _bonferroni_correction([t_pval, u_pval])

    # FDR for 2 tests
    p_values = [t_pval, u_pval]
    fdr_corrections = _fdr_correction(p_values)
    t_pval_fdr = fdr_corrections[0]
    u_pval_fdr = fdr_corrections[1]

    return HypothesisTestResult(
        season=season or "all",
        feature_name=feature_name,
        group1_label=f"high_water_>{threshold:.4f}",
        group2_label=f"low_water_<{threshold:.4f}",
        threshold_percentile=threshold_percentile,
        ttest_statistic=float(t_stat),
        ttest_p_value=float(t_pval),
        ttest_cohens_d=float(cohens_d),
        mannwhitney_statistic=float(u_stat),
        mannwhitney_p_value=float(u_pval),
        mannwhitney_rank_biserial=float(r_rb),
        ttest_p_value_bonferroni=float(t_pval_bonf),
        ttest_p_value_fdr=float(t_pval_fdr),
        mannwhitney_p_value_bonferroni=float(u_pval_bonf),
        mannwhitney_p_value_fdr=float(u_pval_fdr),
        shapiro_p_value_group1=float(shapiro_p1),
        shapiro_p_value_group2=float(shapiro_p2),
        n_group1=len(group_high),
        n_group2=len(group_low),
        n_missing_group1=int(data[data[target_column] >= threshold][feature_name].isna().sum()),
        n_missing_group2=int(data[data[target_column] < threshold][feature_name].isna().sum()),
    )


def soil_saturation_event_crosstab(
    df: pd.DataFrame,
    soil_saturation_column: str = "soil_saturation_index",
    event_column: str = "event_occurred",
    season: str | None = None,
) -> CrosstabResult | None:
    """
    Cross-tabulate soil saturation levels (quartiles) vs event occurrence (contingency table).

    Args:
        df: DataFrame with soil_saturation_column, event_column, and optionally 'season'
        soil_saturation_column: Name of soil saturation feature
        event_column: Name of event binary column (0/1)
        season: Season name. If None, computes across all data.

    Returns:
        CrosstabResult with contingency table and chi-square test results, or None if insufficient data.

    Notes:
        - Uses pair-wise deletion for missing values.
        - Creates 4 saturation levels using quartiles (Q1, Q2-Q3, Q4).
        - Reports both uncorrected and Bonferroni/FDR-corrected p-values.
    """
    if soil_saturation_column not in df.columns:
        return None
    if event_column not in df.columns:
        return None

    # Filter by season
    data = df.copy()
    if season is not None:
        if "season" not in data.columns:
            raise ValueError(
                "DataFrame must have 'season' column when season parameter is specified"
            )
        data = data[data["season"] == season]

    if len(data) == 0:
        return None

    # Pair-wise deletion
    clean = data[[soil_saturation_column, event_column]].dropna()

    if len(clean) < 4:
        return None

    # Binarize event (handle various input formats)
    event_binary = clean[event_column].astype(str).isin(["1", "True", "true", 1, True])

    # Categorize soil saturation into quartiles
    soil_sat_cat = pd.qcut(
        clean[soil_saturation_column], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )

    # Build contingency table
    cont_table = pd.crosstab(soil_sat_cat, event_binary)

    # Chi-square test
    chi2_result = cast(
        tuple[float, float, Any, Any],
        stats.chi2_contingency(cont_table),
    )
    chi2 = chi2_result[0]
    chi2_pval = chi2_result[1]

    # Cramér's V effect size
    n = int(np.asarray(cont_table.to_numpy(dtype=float).sum()).item())
    rows, cols = cont_table.shape
    min_dim = max(1, min(int(rows) - 1, int(cols) - 1))
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0

    # Multiple testing corrections
    chi2_pval_bonf = _bonferroni_correction([chi2_pval])[0]  # Single chi-square test
    chi2_pval_fdr = chi2_pval  # Single test, FDR = p-value

    # Convert contingency table to dict format
    cont_dict = {}
    for row_label in cont_table.index:
        cont_dict[str(row_label)] = {
            str(col): int(cast(float, cont_table.loc[row_label, col])) for col in cont_table.columns
        }

    return CrosstabResult(
        season=season or "all",
        row_variable=soil_saturation_column,
        col_variable=event_column,
        contingency_table=cont_dict,
        chi2_statistic=float(chi2),
        chi2_p_value=float(chi2_pval),
        cramers_v=float(cramers_v),
        chi2_p_value_bonferroni=float(chi2_pval_bonf),
        chi2_p_value_fdr=float(chi2_pval_fdr),
        n_samples=int(n),
    )


def analyze_onset_error_distribution(
    onset_errors_hours: Sequence[float | int],
    season: str | None = None,
) -> OnsetErrorDistribution | None:
    """
    Analyze distribution of onset errors (predicted event start vs actual start).

    Args:
        onset_errors_hours: List of onset error values in hours
        season: Season name for labeling

    Returns:
        OnsetErrorDistribution with statistical summary, or None if insufficient data.
    """
    if not onset_errors_hours or len(onset_errors_hours) == 0:
        return None

    errors = np.asarray(onset_errors_hours, dtype=np.float64)
    # Remove NaN and inf
    errors = errors[~(np.isnan(errors) | np.isinf(errors))]

    if len(errors) < 1:
        return None

    return OnsetErrorDistribution(
        season=season or "all",
        min_hours=float(np.min(errors)),
        max_hours=float(np.max(errors)),
        mean_hours=float(np.mean(errors)),
        median_hours=float(np.median(errors)),
        std_hours=float(np.std(errors)) if len(errors) > 1 else 0.0,
        p10_hours=float(np.percentile(errors, 10)),
        p25_hours=float(np.percentile(errors, 25)),
        p75_hours=float(np.percentile(errors, 75)),
        p90_hours=float(np.percentile(errors, 90)),
        n_errors=len(errors),
    )


class StatisticalAnalyzer:
    """
    Orchestrates comprehensive statistical analysis of water level data.

    Performs seasonal breakdown of:
    - Lag feature correlations
    - Hypothesis tests for feature contributions
    - Soil saturation vs event contingency
    - Onset error distributions

    Notes:
        - Assumes input data is already normalized (scaled).
        - Handles missing values via pair-wise deletion with warnings.
        - Reports both parametric and non-parametric tests.
        - Includes multiple testing corrections (Bonferroni, FDR).
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "unnamed"):
        """
        Initialize analyzer with preprocessed data.

        Args:
            df: DataFrame with 'timestamp', 'season', normalized features, lag features, event labels
            dataset_name: Name for reporting purposes
        """
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.warnings: list[str] = []
        self.notes: list[str] = []

        # Validate required columns
        if "timestamp" not in self.df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        # Add season if missing
        if "season" not in self.df.columns:
            self.df = self._add_temporal_columns()

        # Check for normalization indicator (columns should be in [-5, 5] range roughly for zscore)
        self._check_normalization()

    def _add_temporal_columns(self) -> pd.DataFrame:
        """Add season column from timestamp."""
        df = self.df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        month = df["timestamp"].dt.month
        df["season"] = pd.Series(
            np.select(
                [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
                ["winter", "spring", "summer"],
                default="autumn",
            ),
            index=df.index,
        )
        return df

    def _check_normalization(self) -> None:
        """Check if data appears to be normalized by examining column ranges."""
        numeric_cols = self.df.select_dtypes(include=["float64", "float32"]).columns

        # Exclude timestamp-derived columns
        feature_cols = [col for col in numeric_cols if col not in ["timestamp", "year"]]

        if len(feature_cols) == 0:
            return

        # For zscore normalization, most values should be in [-4, 4]
        # For minmax, values should be in [0, 1]
        sample_cols = feature_cols[: min(10, len(feature_cols))]

        for col in sample_cols:
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                continue

            col_min = col_data.min()
            col_max = col_data.max()
            col_mean = col_data.mean()
            col_std = col_data.std()

            # Check if likely zscore normalized (mean ~0, std ~1)
            if abs(col_mean) < 0.5 and 0.5 < col_std < 2.0:
                continue  # Likely normalized

            # Check if likely minmax normalized (range [0, 1])
            if col_min >= -0.1 and col_max <= 1.1:
                continue  # Likely normalized

            # Otherwise, warn
            self.warnings.append(
                f"Column '{col}' may not be normalized: mean={col_mean:.4f}, std={col_std:.4f}, "
                f"min={col_min:.4f}, max={col_max:.4f}. Ensure data is zscore/minmax/robust scaled."
            )
            break  # Only warn once

    def analyze_lags_by_season(
        self,
        target_column: str = "water_level_m",
        lag_columns: list[str] | None = None,
    ) -> list[SeasonalLagCorrelation]:
        """
        Compute lag feature correlations for each season.

        Args:
            target_column: Water level column name
            lag_columns: Optional list of lag column names. If None, auto-detect.

        Returns:
            List of SeasonalLagCorrelation results.
        """
        all_results: list[SeasonalLagCorrelation] = []

        seasons = self.df["season"].unique()
        for season in sorted(seasons):
            season_results = compute_lag_correlations(
                self.df,
                target_column=target_column,
                season=season,
                lag_columns=lag_columns,
            )
            all_results.extend(season_results)

        # Also compute across all data
        all_data_results = compute_lag_correlations(
            self.df,
            target_column=target_column,
            season=None,
            lag_columns=lag_columns,
        )
        all_results.extend(all_data_results)

        return all_results

    def analyze_group_differences(
        self,
        features_to_test: list[str],
        target_column: str = "water_level_m",
        threshold_percentile: float = 95.0,
    ) -> list[HypothesisTestResult]:
        """
        Test feature differences between high and low water level groups (per season).

        Args:
            features_to_test: List of feature names to compare
            target_column: Water level column name
            threshold_percentile: Cutoff for "high water" label

        Returns:
            List of HypothesisTestResult objects.
        """
        all_results: list[HypothesisTestResult] = []

        seasons = sorted(self.df["season"].unique())
        for season in seasons:
            for feature in features_to_test:
                result = compare_groups_by_threshold(
                    self.df,
                    feature_name=feature,
                    target_column=target_column,
                    threshold_percentile=threshold_percentile,
                    season=season,
                )
                if result:
                    all_results.append(result)

        # Also test across all data
        for feature in features_to_test:
            result = compare_groups_by_threshold(
                self.df,
                feature_name=feature,
                target_column=target_column,
                threshold_percentile=threshold_percentile,
                season=None,
            )
            if result:
                all_results.append(result)

        return all_results

    def analyze_soil_saturation(
        self,
        soil_saturation_column: str = "soil_saturation_index",
        event_column: str = "event_occurred",
    ) -> list[CrosstabResult]:
        """
        Analyze contingency between soil saturation and event occurrence (per season).

        Args:
            soil_saturation_column: Soil saturation feature name
            event_column: Event binary column name

        Returns:
            List of CrosstabResult objects.
        """
        all_results: list[CrosstabResult] = []

        seasons = sorted(self.df["season"].unique())
        for season in seasons:
            result = soil_saturation_event_crosstab(
                self.df,
                soil_saturation_column=soil_saturation_column,
                event_column=event_column,
                season=season,
            )
            if result:
                all_results.append(result)

        # Also analyze across all data
        result = soil_saturation_event_crosstab(
            self.df,
            soil_saturation_column=soil_saturation_column,
            event_column=event_column,
            season=None,
        )
        if result:
            all_results.append(result)

        return all_results

    def analyze_onset_errors(
        self,
        onset_errors_by_season: Mapping[str, Sequence[float | int]],
    ) -> list[OnsetErrorDistribution]:
        """
        Analyze onset error distributions per season.

        Args:
            onset_errors_by_season: Dict mapping season name to list of onset errors in hours

        Returns:
            List of OnsetErrorDistribution objects.
        """
        all_results: list[OnsetErrorDistribution] = []

        for season, errors in onset_errors_by_season.items():
            result = analyze_onset_error_distribution(errors, season=season)
            if result:
                all_results.append(result)

        return all_results

    def generate_statistical_summary(
        self,
        target_column: str = "water_level_m",
        event_column: str = "event_occurred",
        soil_saturation_column: str | None = "soil_saturation_index",
        features_to_test: list[str] | None = None,
        lag_columns: list[str] | None = None,
        onset_errors_by_season: Mapping[str, Sequence[float | int]] | None = None,
        threshold_percentile: float = 95.0,
    ) -> StatisticalSummary:
        """
        Run all statistical analyses and compile into a unified summary.

        Args:
            target_column: Water level column
            event_column: Event binary column
            soil_saturation_column: Soil saturation feature
            features_to_test: List of features for hypothesis tests. If None, auto-select key features.
            lag_columns: List of lag columns for correlation analysis
            onset_errors_by_season: Dict of onset errors by season from evaluation
            threshold_percentile: Percentile cutoff for high water definition

        Returns:
            StatisticalSummary object with all results.
        """
        # Auto-select features if not provided
        if features_to_test is None:
            features_to_test = [
                col
                for col in self.df.columns
                if col.startswith(("rainfall", "temperature", "pressure", "soil_saturation"))
                and "_lag_" not in col
            ]

        # Compute timestamp range
        ts_col = pd.to_datetime(self.df["timestamp"])
        ts_min = ts_col.min().isoformat()
        ts_max = ts_col.max().isoformat()

        # Run all analyses
        lag_corr = self.analyze_lags_by_season(target_column=target_column, lag_columns=lag_columns)
        hyp_tests = self.analyze_group_differences(
            features_to_test,
            target_column=target_column,
            threshold_percentile=threshold_percentile,
        )
        crosstabs = self.analyze_soil_saturation(
            soil_saturation_column=soil_saturation_column or "soil_saturation_index",
            event_column=event_column,
        )
        onset_errors = []
        if onset_errors_by_season:
            onset_errors = self.analyze_onset_errors(onset_errors_by_season)

        return StatisticalSummary(
            dataset_name=self.dataset_name,
            timestamp_range=(ts_min, ts_max),
            n_total_rows=len(self.df),
            lag_correlations=lag_corr,
            hypothesis_tests=hyp_tests,
            crosstab_results=crosstabs,
            onset_error_distributions=onset_errors,
            warnings=self.warnings,
            notes=self.notes,
        )
