"""Describes statistical analysis of the data."""

from src.analysis.schemas import (
    CrosstabResult,
    HypothesisTestResult,
    OnsetErrorDistribution,
    SeasonalLagCorrelation,
    StatisticalSummary,
)
from src.analysis.statistical_analyzer import (
    StatisticalAnalyzer,
    analyze_onset_error_distribution,
    compare_groups_by_threshold,
    compute_lag_correlations,
    soil_saturation_event_crosstab,
)

__all__ = [
    "CrosstabResult",
    "HypothesisTestResult",
    "OnsetErrorDistribution",
    "SeasonalLagCorrelation",
    "StatisticalAnalyzer",
    "StatisticalSummary",
    "analyze_onset_error_distribution",
    "compare_groups_by_threshold",
    "compute_lag_correlations",
    "soil_saturation_event_crosstab",
]
