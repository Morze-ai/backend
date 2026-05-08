"""Feature importance utilities based on SHAP values."""

from __future__ import annotations

import pandas as pd

from src.explain.utils import mean_absolute_shap_importance


def rank_features(
    shap_values,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Ranks features by mean absolute SHAP importance.
    """

    importance = mean_absolute_shap_importance(shap_values)

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    result = result.sort_values(
        by="importance",
        ascending=False,
    )

    result = result.reset_index(drop=True)

    return result


def top_k_features(
    importance_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """
    Returns the top-k most important features.
    """

    return importance_df.head(k)
