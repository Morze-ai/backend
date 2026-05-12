"""Tests for ERA5 processor utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    import xarray as xr
else:
    try:
        import xarray as xr
    except ImportError:
        xr = None  # type: ignore[assignment]

from src.data.era5_processor import (
    process_pressure_folder,
    process_sst_folder,
    process_wind_folder,
    process_year,
)


def create_era5_netcdf(path: Path, var_name: str, data: np.ndarray) -> None:
    """Helper to create a test ERA5 netCDF file."""
    times = pd.date_range("2021-01-01", periods=data.shape[0], freq="6h")
    ds = xr.Dataset(
        {var_name: (["valid_time", "latitude", "longitude"], data)},
        coords={
            "valid_time": times,
            "latitude": [54.2, 54.5],
            "longitude": [18.3, 18.9],
        },
    )
    ds.to_netcdf(path)


@pytest.mark.skipif(xr is None, reason="xarray not installed")
class TestWindProcessor:
    """Tests for wind data processing."""

    def test_process_wind_folder_valid(self, tmp_path: Path) -> None:
        """Test processing valid wind u/v components."""
        u_data = np.random.randn(10, 2, 2) * 5
        v_data = np.random.randn(10, 2, 2) * 5

        u_path = tmp_path / "u10.nc"
        v_path = tmp_path / "v10.nc"

        create_era5_netcdf(u_path, "u10", u_data)
        create_era5_netcdf(v_path, "v10", v_data)

        # Create combined file
        combined_path = tmp_path / "wind_combined.nc"
        times = pd.date_range("2021-01-01", periods=10, freq="6h")
        ds = xr.Dataset(
            {
                "u10": (["valid_time", "latitude", "longitude"], u_data),
                "v10": (["valid_time", "latitude", "longitude"], v_data),
            },
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds.to_netcdf(combined_path)

        result = process_wind_folder(combined_path)

        assert not result.empty
        assert "timestamp" in result.columns
        assert "wind_u" in result.columns
        assert "wind_v" in result.columns
        assert "wind_speed_ms" in result.columns
        assert "wind_direction_deg" in result.columns

    def test_process_wind_missing_components(self, tmp_path: Path) -> None:
        """Test that missing u or v component returns empty DataFrame."""
        u_data = np.random.randn(5, 2, 2)
        u_path = tmp_path / "u_only.nc"

        times = pd.date_range("2021-01-01", periods=5, freq="6h")
        ds = xr.Dataset(
            {"u10": (["valid_time", "latitude", "longitude"], u_data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds.to_netcdf(u_path)

        result = process_wind_folder(u_path)

        assert result.empty


@pytest.mark.skipif(xr is None, reason="xarray not installed")
class TestPressureProcessor:
    """Tests for pressure data processing."""

    def test_process_pressure_folder_valid(self, tmp_path: Path) -> None:
        """Test processing valid mean sea level pressure."""
        msl_data = np.full((8, 2, 2), 1013.25)  # hPa

        p_path = tmp_path / "pressure.nc"
        create_era5_netcdf(p_path, "msl", msl_data)

        result = process_pressure_folder(p_path)

        assert not result.empty
        assert "timestamp" in result.columns
        assert "pressure_hpa" in result.columns
        # Value should be ~1013.25 hPa (or divided from Pa if stored in Pa)
        assert result["pressure_hpa"].notna().all()

    def test_process_pressure_missing(self, tmp_path: Path) -> None:
        """Test that missing pressure variable returns empty DataFrame."""
        other_data = np.random.randn(5, 2, 2)

        p_path = tmp_path / "no_pressure.nc"
        times = pd.date_range("2021-01-01", periods=5, freq="6h")
        ds = xr.Dataset(
            {"other_var": (["valid_time", "latitude", "longitude"], other_data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds.to_netcdf(p_path)

        result = process_pressure_folder(p_path)

        assert result.empty


@pytest.mark.skipif(xr is None, reason="xarray not installed")
class TestSSTProcessor:
    """Tests for sea surface temperature data processing."""

    def test_process_sst_folder_valid(self, tmp_path: Path) -> None:
        """Test processing valid SST data."""
        sst_data = np.full((8, 2, 2), 280.0)  # Kelvin

        sst_path = tmp_path / "sst.nc"
        create_era5_netcdf(sst_path, "sst", sst_data)

        result = process_sst_folder(sst_path)

        assert not result.empty
        assert "timestamp" in result.columns
        assert "sea_surface_temperature_c" in result.columns
        # Should be converted from K to C (280K - 273.15 = ~6.85C)
        assert result["sea_surface_temperature_c"].notna().all()
        assert 0 < result["sea_surface_temperature_c"].mean() < 20

    def test_process_sst_all_nan(self, tmp_path: Path) -> None:
        """Test processing SST with all NaN values (coastal area)."""
        sst_data = np.full((8, 2, 2), np.nan)

        sst_path = tmp_path / "sst_nan.nc"
        create_era5_netcdf(sst_path, "sst", sst_data)

        result = process_sst_folder(sst_path)

        # Should return non-empty frame even with all NaN (caller handles filling)
        # Actually, with all NaN it might return empty after resampling
        # This tests the edge case
        assert isinstance(result, pd.DataFrame)

    def test_process_sst_missing(self, tmp_path: Path) -> None:
        """Test that missing SST variable returns empty DataFrame."""
        other_data = np.random.randn(5, 2, 2)

        sst_path = tmp_path / "no_sst.nc"
        times = pd.date_range("2021-01-01", periods=5, freq="6h")
        ds = xr.Dataset(
            {"other_var": (["valid_time", "latitude", "longitude"], other_data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds.to_netcdf(sst_path)

        result = process_sst_folder(sst_path)

        assert result.empty


@pytest.mark.skipif(xr is None, reason="xarray not installed")
class TestProcessYear:
    """Tests for yearly ERA5 processing orchestration."""

    def test_process_year_all_variables(self, tmp_path: Path) -> None:
        """Test processing a complete year with wind, pressure, and SST."""
        u_data = np.random.randn(8, 2, 2) * 3
        v_data = np.random.randn(8, 2, 2) * 3
        msl_data = np.full((8, 2, 2), 1013.25)
        sst_data = np.full((8, 2, 2), 280.0)

        times = pd.date_range("2021-01-01", periods=8, freq="6h")

        # Create wind file
        wind_path = tmp_path / "2021" / "wind" / "data.nc"
        wind_path.parent.mkdir(parents=True, exist_ok=True)
        ds_wind = xr.Dataset(
            {
                "u10": (["valid_time", "latitude", "longitude"], u_data),
                "v10": (["valid_time", "latitude", "longitude"], v_data),
            },
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds_wind.to_netcdf(wind_path)

        # Create pressure file
        p_path = tmp_path / "2021" / "pressure" / "data.nc"
        p_path.parent.mkdir(parents=True, exist_ok=True)
        ds_p = xr.Dataset(
            {"msl": (["valid_time", "latitude", "longitude"], msl_data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds_p.to_netcdf(p_path)

        # Create SST file
        sst_path = tmp_path / "2021" / "sea_surface_temperature" / "data.nc"
        sst_path.parent.mkdir(parents=True, exist_ok=True)
        ds_sst = xr.Dataset(
            {"sst": (["valid_time", "latitude", "longitude"], sst_data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds_sst.to_netcdf(sst_path)

        # Mock the DATA_ROOT by temporarily patching
        import src.data.era5_processor as ep

        orig_root = ep.DATA_ROOT
        ep.DATA_ROOT = tmp_path

        try:
            result = process_year(2021)

            assert not result.empty
            assert "timestamp" in result.columns
            assert "wind_u" in result.columns
            assert "wind_speed_ms" in result.columns
            assert "pressure_hpa" in result.columns
        finally:
            ep.DATA_ROOT = orig_root
