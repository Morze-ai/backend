"""Data structures for statistical analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SeasonalLagCorrelation:
    """Correlation between a lag feature and target variable within a season."""

    season: str
    lag_hours: int
    feature_name: str
    target_column: str
    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    n_samples: int
    n_missing: int


@dataclass(slots=True)
class HypothesisTestResult:
    """Result of comparing two groups (high vs low water level)."""

    season: str
    feature_name: str
    group1_label: str
    group2_label: str
    threshold_percentile: float
    # Parametric test (t-test)
    ttest_statistic: float
    ttest_p_value: float
    ttest_cohens_d: float
    # Non-parametric test (Mann-Whitney U)
    mannwhitney_statistic: float
    mannwhitney_p_value: float
    mannwhitney_rank_biserial: float
    # Multiple testing corrections for ttest
    ttest_p_value_bonferroni: float
    ttest_p_value_fdr: float
    # Multiple testing corrections for Mann-Whitney
    mannwhitney_p_value_bonferroni: float
    mannwhitney_p_value_fdr: float
    # Normality test
    shapiro_p_value_group1: float
    shapiro_p_value_group2: float
    # Sample sizes
    n_group1: int
    n_group2: int
    n_missing_group1: int
    n_missing_group2: int


@dataclass(slots=True)
class CrosstabResult:
    """Contingency table analysis (soil saturation vs event occurrence)."""

    season: str
    row_variable: str
    col_variable: str
    contingency_table: dict[str, dict[str, int]]  # row_label -> {col_label -> count}
    chi2_statistic: float
    chi2_p_value: float
    cramers_v: float
    # Bonferroni & FDR corrections
    chi2_p_value_bonferroni: float
    chi2_p_value_fdr: float
    n_samples: int


@dataclass(slots=True)
class OnsetErrorDistribution:
    """Statistical distribution of onset errors (predicted vs actual event start)."""

    season: str
    min_hours: float
    max_hours: float
    mean_hours: float
    median_hours: float
    std_hours: float
    p10_hours: float
    p25_hours: float
    p75_hours: float
    p90_hours: float
    n_errors: int


@dataclass(slots=True)
class StatisticalSummary:
    """Container for all statistical analysis results."""

    dataset_name: str
    timestamp_range: tuple[str, str]  # (start_iso, end_iso)
    n_total_rows: int
    # Results by season
    lag_correlations: list[SeasonalLagCorrelation]
    hypothesis_tests: list[HypothesisTestResult]
    crosstab_results: list[CrosstabResult]
    onset_error_distributions: list[OnsetErrorDistribution]
    # Overall summaries
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "timestamp_range": self.timestamp_range,
            "n_total_rows": self.n_total_rows,
            "lag_correlations": [self._dataclass_to_dict(lc) for lc in self.lag_correlations],
            "hypothesis_tests": [self._dataclass_to_dict(ht) for ht in self.hypothesis_tests],
            "crosstab_results": [self._dataclass_to_dict(cr) for cr in self.crosstab_results],
            "onset_error_distributions": [
                self._dataclass_to_dict(oed) for oed in self.onset_error_distributions
            ],
            "warnings": self.warnings,
            "notes": self.notes,
        }

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
        """Recursively convert dataclass to dict, handling np.nan and np.inf."""
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if isinstance(value, (np.floating, float)):
                    if np.isnan(value) or np.isinf(value):
                        result[field_name] = None
                    else:
                        result[field_name] = float(value)
                elif isinstance(value, np.integer):
                    result[field_name] = int(value)
                elif isinstance(value, dict):
                    result[field_name] = value
                else:
                    result[field_name] = value
            return result
        return obj
