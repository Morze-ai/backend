"""Process ERA5 netCDF files into hourly CSVs with wind, pressure, and SST.

Provides a minimal, robust implementation that attempts to locate common
variable names and produce a merged hourly file for 2021-2025 by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.netcdf_loader import (
    concat_yearly_frames,
    load_netcdf_to_dataframe,
    resample_to_hourly,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "era5_hourly_full.csv"


def _select_var(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def process_wind_folder(nc_path: Path) -> pd.DataFrame:
    df = load_netcdf_to_dataframe(nc_path)

    u_cands = ["u10", "u_component_of_wind_10m", "u"]
    v_cands = ["v10", "v_component_of_wind_10m", "v"]

    u_col = _select_var(df, u_cands)
    v_col = _select_var(df, v_cands)

    if u_col is None or v_col is None:
        # Try to guess from variable suffixes
        # If missing, return empty DataFrame
        return pd.DataFrame()

    df = df[["timestamp", u_col, v_col]].dropna(subset=["timestamp"])  # keep rows with timestamps
    df = df.rename(columns={u_col: "wind_u", v_col: "wind_v"})

    # compute speed and direction
    df["wind_speed_ms"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)
    # direction degrees, meteorological: atan2(u,v) -> arctan2(u, v) convert to degrees and normalize
    df["wind_direction_deg"] = (np.degrees(np.arctan2(df["wind_u"], df["wind_v"])) + 360) % 360

    df = resample_to_hourly(df)
    return df


def process_pressure_folder(nc_path: Path) -> pd.DataFrame:
    df = load_netcdf_to_dataframe(nc_path)
    # common names
    p_cands = ["mean_sea_level_pressure", "msl", "pressure"]
    p_col = _select_var(df, p_cands)
    if p_col is None:
        return pd.DataFrame()

    df = df[["timestamp", p_col]].rename(columns={p_col: "pressure"})
    # convert to hPa if values look like Pa (>2000)
    if df["pressure"].median() > 2000:
        df["pressure_hpa"] = df["pressure"] / 100.0
    else:
        df["pressure_hpa"] = df["pressure"]

    df = df[["timestamp", "pressure_hpa"]]
    df = resample_to_hourly(df)
    return df


def process_sst_folder(nc_path: Path) -> pd.DataFrame:
    df = load_netcdf_to_dataframe(nc_path)
    sst_cands = ["sea_surface_temperature", "sst"]
    sst_col = _select_var(df, sst_cands)
    if sst_col is None:
        return pd.DataFrame()

    df = df[["timestamp", sst_col]].rename(columns={sst_col: "sea_surface_temperature_c"})
    # if values are Kelvin (>100), convert to C
    if df["sea_surface_temperature_c"].median() > 100:
        df["sea_surface_temperature_c"] = df["sea_surface_temperature_c"] - 273.15

    df = resample_to_hourly(df)
    return df


def process_year(year: int) -> pd.DataFrame:
    parts = []
    base = DATA_ROOT / str(year)
    wind_folder = base / "wind" / "data.nc"
    pressure_folder = base / "pressure" / "data.nc"
    sst_folder = base / "sea_surface_temperature" / "data.nc"

    if wind_folder.exists():
        wind_df = process_wind_folder(wind_folder)
        if not wind_df.empty:
            parts.append(wind_df)

    if pressure_folder.exists():
        p_df = process_pressure_folder(pressure_folder)
        if not p_df.empty:
            parts.append(p_df)

    if sst_folder.exists():
        sst_df = process_sst_folder(sst_folder)
        if not sst_df.empty:
            parts.append(sst_df)

    if not parts:
        return pd.DataFrame()

    # merge on timestamp
    merged = parts[0]
    for other in parts[1:]:
        merged = pd.merge(merged, other, on="timestamp", how="outer")

    # prefer inner-like coverage by forward/backfilling numeric gaps later in pipeline
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def process_all_years(start_year: int = 2021, end_year: int = 2025) -> pd.DataFrame:
    frames = []
    for y in range(start_year, end_year + 1):
        df = process_year(y)
        if not df.empty:
            frames.append(df)

    full = concat_yearly_frames(frames)
    if not full.empty:
        parent = OUTPUT_PATH.parent
        parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(OUTPUT_PATH, index=False)
    return full


if __name__ == "__main__":
    print("Processing ERA5 raw netCDF folders to hourly CSV...")
    out = process_all_years()
    print("Wrote:", OUTPUT_PATH if not out.empty else "(no data found)")
