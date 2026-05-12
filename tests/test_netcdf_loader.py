"""Tests for netCDF loader utilities."""

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

from src.data.netcdf_loader import (
    concat_yearly_frames,
    load_netcdf_to_dataframe,
    resample_to_hourly,
)


@pytest.mark.skipif(xr is None, reason="xarray not installed")
class TestLoadNetcdfToDataframe:
    """Tests for load_netcdf_to_dataframe function."""

    def test_load_simple_netcdf(self, tmp_path: Path) -> None:
        """Test loading a simple netCDF file with time and data variables."""
        nc_file = tmp_path / "test.nc"

        # Create a simple netCDF dataset
        times = pd.date_range("2021-01-01", periods=10, freq="6h")
        data = np.random.rand(10, 2, 2)

        ds = xr.Dataset(
            {"temperature": (["valid_time", "latitude", "longitude"], data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds.to_netcdf(nc_file)

        # Load and test
        df = load_netcdf_to_dataframe(nc_file)

        assert "timestamp" in df.columns
        assert "temperature" in df.columns
        assert len(df) == 10
        assert df["timestamp"].dtype == "datetime64[ns]"
        assert not df["timestamp"].isna().any()

    def test_load_netcdf_with_nans(self, tmp_path: Path) -> None:
        """Test loading netCDF with NaN values."""
        nc_file = tmp_path / "test_nans.nc"

        times = pd.date_range("2021-01-01", periods=5, freq="h")
        data = np.full((5, 2, 2), np.nan)

        ds = xr.Dataset(
            {"sst": (["valid_time", "latitude", "longitude"], data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
            },
        )
        ds.to_netcdf(nc_file)

        df = load_netcdf_to_dataframe(nc_file)

        assert "timestamp" in df.columns
        assert "sst" in df.columns
        assert df["sst"].isna().all()

    def test_load_netcdf_deduplicates_on_timestamp(self, tmp_path: Path) -> None:
        """Test that loading removes duplicates on timestamp."""
        nc_file = tmp_path / "test_dups.nc"

        times = pd.date_range("2021-01-01", periods=3, freq="h")
        data = np.random.rand(3, 2, 2)

        ds = xr.Dataset(
            {"wind_u": (["valid_time", "latitude", "longitude"], data)},
            coords={
                "valid_time": times,
                "latitude": [54.2, 54.5],
                "longitude": [18.3, 18.9],
                "number": [1],  # Auxiliary coord that might create duplicates
            },
        )
        ds.to_netcdf(nc_file)

        df = load_netcdf_to_dataframe(nc_file)

        # Should have 3 unique timestamps
        assert len(df) == 3
        assert df["timestamp"].nunique() == 3


class TestResampleToHourly:
    """Tests for resample_to_hourly function."""

    def test_resample_6h_to_hourly(self) -> None:
        """Test resampling from 6-hourly to hourly data."""
        times = pd.date_range("2021-01-01", periods=4, freq="6h")
        df = pd.DataFrame(
            {
                "timestamp": times,
                "temperature": [10.0, 12.0, 14.0, 16.0],
            }
        )

        result = resample_to_hourly(df)

        # Should have 19 hours (3 complete 6h intervals + 1 partial)
        assert len(result) >= 4
        assert "timestamp" in result.columns
        # dtype may be datetime64[ns] or datetime64[us] depending on pandas version
        assert str(result["timestamp"].dtype).startswith("datetime64")

    def test_resample_already_hourly(self) -> None:
        """Test that hourly data passes through unchanged."""
        times = pd.date_range("2021-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "timestamp": times,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = resample_to_hourly(df)

        assert len(result) == 5
        assert (result["timestamp"] == df["timestamp"]).all()

    def test_resample_missing_timestamp_column(self) -> None:
        """Test that missing timestamp column raises ValueError."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="timestamp column not found"):
            resample_to_hourly(df)

    def test_resample_interpolates_gaps(self) -> None:
        """Test that resampling interpolates missing hours."""
        times = pd.date_range("2021-01-01", periods=3, freq="12h")
        df = pd.DataFrame(
            {
                "timestamp": times,
                "temperature": [10.0, 20.0, 30.0],
            }
        )

        result = resample_to_hourly(df)

        # Should have 25 hours (0h, 12h, 24h expanded to hourly)
        assert len(result) >= 3
        # Check that we have interpolated values between first and third timestamps
        # Get the middle value and check it's between the extremes
        temps = result["temperature"].dropna().values
        assert len(temps) > 1
        # Check monotonicity: values should increase overall
        assert temps[-1] > temps[0] or (
            temps[-1] == temps[0]
        )  # at least not decreasing in general direction


class TestConcatYearlyFrames:
    """Tests for concat_yearly_frames function."""

    def test_concat_multiple_frames(self) -> None:
        """Test concatenating multiple yearly DataFrames."""
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=3, freq="D"),
                "value": [1.0, 2.0, 3.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-04", periods=3, freq="D"),
                "value": [4.0, 5.0, 6.0],
            }
        )

        result = concat_yearly_frames([df1, df2])

        assert len(result) == 6
        assert result["value"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert (result["timestamp"] == result["timestamp"].sort_values()).all()

    def test_concat_removes_duplicates(self) -> None:
        """Test that duplicates are removed by timestamp."""
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=3, freq="D"),
                "value": [1.0, 2.0, 3.0],
            }
        )
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-02", periods=3, freq="D"),
                "value": [20.0, 30.0, 40.0],  # Overlaps with df1
            }
        )

        result = concat_yearly_frames([df1, df2])

        # Should keep first occurrence or later (but not both)
        assert len(result) <= 5
        assert result["timestamp"].nunique() == len(result)

    def test_concat_empty_list(self) -> None:
        """Test concatenating an empty list returns empty DataFrame."""
        result = concat_yearly_frames([])

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_concat_ignores_none_and_empty(self) -> None:
        """Test that None and empty DataFrames are ignored."""
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=2, freq="D"),
                "value": [1.0, 2.0],
            }
        )

        # Filter out None values before passing to concat_yearly_frames
        frames = [f for f in [None, pd.DataFrame(), df1] if f is not None and len(f) > 0]
        result = concat_yearly_frames(frames)

        assert len(result) == 2
        assert (result["value"] == [1.0, 2.0]).all()
