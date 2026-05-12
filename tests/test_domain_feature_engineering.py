"""Tests for domain-specific feature engineering helpers."""

import pandas as pd
import pytest

from src.data.feature_engineering import (
    calculate_rain_sums,
    calculate_soil_saturation,
    calculate_temp_delta,
    calculate_temp_mean,
    calculate_thaw_flag,
    calculate_wind_features,
    engineer_features,
)


@pytest.fixture
def sample_weather_data():
    """Creates a sample weather dataframe for testing."""
    data = {
        "timestamp": pd.date_range(start="2021-01-01", periods=100, freq="h"),
        "rainfall_mm": [1.0] * 100,
        "temperature_c": [float(i % 20 - 10) for i in range(100)],
        "wind_speed_ms": [5.0] * 100,
        "wind_direction_deg": [180.0] * 100,
    }
    return pd.DataFrame(data)


def test_calculate_rain_sums(sample_weather_data):
    df = calculate_rain_sums(sample_weather_data, windows=[1, 3])

    assert "rain_1h_sum" in df.columns
    assert "rain_3h_sum" in df.columns

    assert df["rain_1h_sum"].iloc[0] == 1.0
    assert df["rain_3h_sum"].iloc[0] == 1.0
    assert df["rain_3h_sum"].iloc[1] == 2.0
    assert df["rain_3h_sum"].iloc[2] == 3.0
    assert df["rain_3h_sum"].iloc[3] == 3.0


def test_calculate_temp_delta(sample_weather_data):
    df = calculate_temp_delta(sample_weather_data, window=24)

    assert "temp_delta_24h" in df.columns
    assert df["temp_delta_24h"].iloc[24] == 4.0
    assert pd.isna(df["temp_delta_24h"].iloc[0])


def test_calculate_thaw_flag(sample_weather_data):
    df = calculate_thaw_flag(sample_weather_data, window=24)

    assert "thaw_flag" in df.columns
    assert df["thaw_flag"].iloc[0] == 0
    assert df["thaw_flag"].iloc[11] == 1

    custom_df = pd.DataFrame({"temperature_c": [1.0, 2.0, 3.0, 4.0]})
    df_custom = calculate_thaw_flag(custom_df, window=2)
    assert (df_custom["thaw_flag"] == 0).all()


def test_calculate_soil_saturation(sample_weather_data):
    df = calculate_soil_saturation(sample_weather_data)

    assert "soil_saturation_index" in df.columns
    assert not df["soil_saturation_index"].isna().any()
    assert df["soil_saturation_index"].iloc[-1] > 0


def test_calculate_temp_mean(sample_weather_data):
    df = calculate_temp_mean(sample_weather_data, window=24)

    assert "temp_mean" in df.columns
    df_20 = calculate_temp_mean(sample_weather_data, window=20)
    assert df_20["temp_mean"].iloc[19] == pytest.approx(-0.5)


def test_calculate_wind_features(sample_weather_data):
    df = calculate_wind_features(sample_weather_data)

    assert "wind_speed" in df.columns
    assert "wind_direction" in df.columns
    assert "wind_u" in df.columns
    assert "wind_v" in df.columns

    assert df["wind_speed"].iloc[0] == 5.0
    assert df["wind_direction"].iloc[0] == 180.0

    assert df["wind_u"].iloc[0] == pytest.approx(0.0, abs=1e-7)
    assert df["wind_v"].iloc[0] == pytest.approx(-5.0)


def test_engineer_features(sample_weather_data):
    df = engineer_features(sample_weather_data)

    expected_cols = [
        "rain_1h_sum",
        "rain_3h_sum",
        "rain_6h_sum",
        "rain_12h_sum",
        "rain_24h_sum",
        "temp_delta_24h",
        "temp_mean",
        "thaw_flag",
        "soil_saturation_index",
        "wind_speed",
        "wind_direction",
        "wind_u",
        "wind_v",
    ]
    for col in expected_cols:
        assert col in df.columns
