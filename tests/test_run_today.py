from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.cli.run_today import fill_feature_fallbacks


def test_fill_feature_fallbacks_replaces_missing_and_non_finite_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z"], utc=True),
            "temperature_c": [np.nan, 7.5],
            "rainfall_mm": [np.nan, 1.2],
            "pressure_hpa": [np.nan, np.inf],
            "humidity_percentage": [np.nan, 80.0],
            "wind_speed": [2.0, np.nan],
            "wind_direction": [90.0, np.nan],
            "temp_delta_24h": [np.nan, np.nan],
            "rainfall_mm_lag_1h": [np.nan, np.nan],
            "rainfall_mm_mean_3h": [np.nan, np.nan],
            "thaw_flag": [np.nan, np.nan],
            "soil_saturation_index": [np.nan, np.nan],
        }
    )

    filled = fill_feature_fallbacks(frame, SimpleNamespace())

    numeric = filled.select_dtypes(include=[float, int]).to_numpy(dtype=float, copy=True)
    assert np.isfinite(numeric).all()
    assert filled["wind_u"].notna().all()
    assert filled["wind_v"].notna().all()
