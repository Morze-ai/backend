import pandas as pd
import pytest

from src.data.feature_engineering import (
    calculate_rain_sums,
    calculate_soil_saturation,
    calculate_temp_delta,
    calculate_thaw_flag,
    engineer_features,
)


@pytest.fixture
def sample_weather_data():
    """Creates a sample weather dataframe for testing."""
    data = {
        "timestamp": pd.date_range(start="2021-01-01", periods=100, freq="h"),
        "rainfall_mm": [1.0] * 100,
        "temperature_c": [float(i % 20 - 10) for i in range(100)],  # -10 to 9 cycling
    }
    return pd.DataFrame(data)


def test_calculate_rain_sums(sample_weather_data):
    df = calculate_rain_sums(sample_weather_data, windows=[1, 3])

    assert "rain_1h_sum" in df.columns
    assert "rain_3h_sum" in df.columns

    # All rainfall is 1.0, so 1h sum should be 1.0, 3h sum should be 3.0 (except first few)
    assert df["rain_1h_sum"].iloc[0] == 1.0
    assert df["rain_3h_sum"].iloc[0] == 1.0
    assert df["rain_3h_sum"].iloc[1] == 2.0
    assert df["rain_3h_sum"].iloc[2] == 3.0
    assert df["rain_3h_sum"].iloc[3] == 3.0


def test_calculate_temp_delta(sample_weather_data):
    df = calculate_temp_delta(sample_weather_data, window=24)

    assert "temp_delta_24h" in df.columns
    # With cycle of 20, at 24h the difference should be predictable
    # cycle: -10, -9, ..., 9 (length 20)
    # i=0: -10, i=24: -10 + (24%20) = -6. Delta = -6 - (-10) = 4
    assert df["temp_delta_24h"].iloc[24] == 4.0
    assert pd.isna(df["temp_delta_24h"].iloc[0])


def test_calculate_thaw_flag(sample_weather_data):
    # Temperature cycle: -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    df = calculate_thaw_flag(sample_weather_data, window=24)

    assert "thaw_flag" in df.columns

    # At i=0, temp is -10. thaw_flag should be 0 (above zero is false)
    assert df["thaw_flag"].iloc[0] == 0

    # At i=11, temp is 1. Previous temps (0-10) were negative. So thaw_flag should be 1
    assert df["thaw_flag"].iloc[11] == 1

    # If we had a long period of positive temps, thaw_flag should eventually become 0
    # but our cycle always has negative temps in the last 24h.
    # Let's test with custom data
    custom_df = pd.DataFrame({"temperature_c": [1.0, 2.0, 3.0, 4.0]})
    df_custom = calculate_thaw_flag(custom_df, window=2)
    assert (df_custom["thaw_flag"] == 0).all()


def test_calculate_soil_saturation(sample_weather_data):
    df = calculate_soil_saturation(sample_weather_data)

    assert "soil_saturation_index" in df.columns
    # Check if it's not all zeros or NaNs
    assert not df["soil_saturation_index"].isna().any()
    # With constant rainfall, it should approach a steady state
    assert df["soil_saturation_index"].iloc[-1] > 0


def test_engineer_features(sample_weather_data):
    df = engineer_features(sample_weather_data)

    expected_cols = [
        "rain_1h_sum",
        "rain_3h_sum",
        "rain_6h_sum",
        "rain_12h_sum",
        "rain_24h_sum",
        "temp_delta_24h",
        "thaw_flag",
        "soil_saturation_index",
    ]
    for col in expected_cols:
        assert col in df.columns
