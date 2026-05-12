"""Helpers to load ERA5 NetCDF files and convert to hourly DataFrame.

This is intentionally lightweight: it attempts to use xarray when available
and falls back with clear error messages if not.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception:  # pragma: no cover - xarray may not be installed in some envs
    xr = None  # type: ignore


def _guess_spatial_dims(ds) -> list[str]:
    candidates = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("y", "x"),
    ]
    for lat, lon in candidates:
        if lat in ds.coords and lon in ds.coords:
            return [lat, lon]
    return []


def load_netcdf_to_dataframe(nc_path: str | Path) -> pd.DataFrame:
    """Open a netCDF file and return a DataFrame with time and variables averaged
    over the spatial domain when spatial dims are present.

    The returned DataFrame uses column `timestamp` for the time index.
    """
    if xr is None:
        raise RuntimeError("xarray is required to read netCDF files. Install xarray.")

    nc_path = Path(nc_path)
    ds = xr.open_dataset(str(nc_path))

    spatial = _guess_spatial_dims(ds)
    ds_mean = ds.mean(dim=spatial, skipna=True) if spatial else ds

    # Convert to dataframe and ensure a timestamp column exists
    df = ds_mean.to_dataframe().reset_index()

    # prefer common time coordinate names used by ERA5-style files
    time_candidates = ["valid_time", "time", "timestamp", "datetime"]
    for t in time_candidates:
        if t in df.columns:
            df = df.rename(columns={t: "timestamp"})
            break

    if "timestamp" not in df.columns and ds_mean.indexes:
        # fallback: try the first index
        try:
            idx_name = next(iter(ds_mean.indexes.keys()))
            df = df.rename(columns={idx_name: "timestamp"})
        except Exception:
            pass

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop duplicated rows that can arise from auxiliary coordinates such as number/expver.
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"])

    return df


def resample_to_hourly(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Resample a DataFrame to hourly timestamps using linear interpolation.

    Keeps numeric columns and interpolates missing values.
    """
    result = df.copy()
    if timestamp_column not in result.columns:
        raise ValueError("timestamp column not found for resampling")

    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    result = result.dropna(subset=[timestamp_column])
    result = result.set_index(timestamp_column, drop=True)
    result = result.sort_index()

    # If already hourly, return as-is (but ensure integer hourly frequency)
    inferred = (
        pd.infer_freq(cast(pd.DatetimeIndex, result.index[:10]))
        if len(result.index) >= 10
        else None
    )
    if inferred and inferred.upper().endswith("H"):
        return result.reset_index()

    # Resample to hourly and interpolate
    numeric = result.select_dtypes(include=[np.number])
    resampled = numeric.resample("h").interpolate(method="time")
    resampled = resampled.reset_index()

    # Keep non-numeric as forward-filled if present
    non_numeric = result.select_dtypes(exclude=[np.number])
    if not non_numeric.empty:
        non_numeric_resampled = non_numeric.resample("h").ffill().reset_index()
        merged = pd.merge(resampled, non_numeric_resampled, on=timestamp_column, how="left")
        return merged

    return resampled


def concat_yearly_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f.copy() for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return combined
