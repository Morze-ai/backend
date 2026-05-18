"""External data clients used by continuous flood-risk evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests

IMGW_STATION_IDS = [12155, 12135, 12100, 12125, 12120, 12200, 12115]


@dataclass(slots=True)
class ClientResult:
    """Simple transport object for client payload + metadata."""

    values: dict[str, float]
    updated_at: datetime | None


class OpenMeteoClient:
    """Fetches near-real-time meteorological values from Open-Meteo."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_latest(self, latitude: float, longitude: float) -> ClientResult:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": (
                    "temperature_2m,relative_humidity_2m,precipitation,"
                    "pressure_msl,wind_speed_10m,wind_direction_10m"
                ),
                "forecast_days": 1,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})

        timestamps = hourly.get("time", [])
        if not timestamps:
            raise ValueError("Open-Meteo response did not include hourly timestamps.")

        latest_index = len(timestamps) - 1
        latest_timestamp = pd.to_datetime(timestamps[latest_index], utc=True, errors="coerce")
        updated_at = latest_timestamp.to_pydatetime() if pd.notna(latest_timestamp) else None

        wind_speed = float(hourly.get("wind_speed_10m", [0.0])[latest_index] or 0.0)
        wind_direction = float(hourly.get("wind_direction_10m", [0.0])[latest_index] or 0.0)

        radians = np.deg2rad(wind_direction)
        wind_u = float(wind_speed * np.sin(radians))
        wind_v = float(wind_speed * np.cos(radians))

        return ClientResult(
            values={
                "temperature_c": float(hourly.get("temperature_2m", [0.0])[latest_index] or 0.0),
                "humidity_percentage": float(
                    hourly.get("relative_humidity_2m", [0.0])[latest_index] or 0.0
                ),
                "rainfall_mm": float(hourly.get("precipitation", [0.0])[latest_index] or 0.0),
                "pressure_hpa": float(hourly.get("pressure_msl", [0.0])[latest_index] or 0.0),
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "wind_u": wind_u,
                "wind_v": wind_v,
            },
            updated_at=updated_at,
        )

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        past_days: int = 10,
        forecast_days: int = 7,
    ) -> pd.DataFrame:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": (
                    "temperature_2m,relative_humidity_2m,precipitation,"
                    "pressure_msl,wind_speed_10m,wind_direction_10m"
                ),
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "UTC",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})

        timestamps = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_c": hourly.get("temperature_2m", []),
                "humidity_percentage": hourly.get("relative_humidity_2m", []),
                "rainfall_mm": hourly.get("precipitation", []),
                "pressure_hpa": hourly.get("pressure_msl", []),
                "wind_speed": hourly.get("wind_speed_10m", []),
                "wind_direction": hourly.get("wind_direction_10m", []),
            }
        )
        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        if not frame.empty:
            radians = np.deg2rad(
                pd.to_numeric(frame["wind_direction"], errors="coerce").fillna(0.0)
            )
            wind_speed = pd.to_numeric(frame["wind_speed"], errors="coerce").fillna(0.0)
            frame["wind_u"] = wind_speed * np.sin(radians)
            frame["wind_v"] = wind_speed * np.cos(radians)

        return frame


class StormglassClient:
    """Fetches recent marine context from Stormglass for water-level proxy updates."""

    def __init__(self, api_key: str | None, timeout_seconds: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def fetch_latest(self, latitude: float, longitude: float) -> ClientResult:
        if not self.api_key:
            raise ValueError("STORMGLASS_API_KEY is not configured.")

        response = requests.get(
            "https://api.stormglass.io/v2/weather/point",
            params={
                "lat": latitude,
                "lng": longitude,
                "params": "waterLevel",
                "source": "sg",
            },
            headers={"Authorization": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        hours = payload.get("hours", [])
        if not hours:
            raise ValueError("Stormglass response did not include hourly data.")

        latest = hours[-1]
        level_payload = latest.get("waterLevel", {})
        if isinstance(level_payload, dict):
            water_level = float(next(iter(level_payload.values()), 0.0) or 0.0)
        else:
            water_level = float(level_payload or 0.0)

        updated_at = pd.to_datetime(latest.get("time"), utc=True, errors="coerce")

        return ClientResult(
            values={"water_level_m": water_level},
            updated_at=updated_at.to_pydatetime() if pd.notna(updated_at) else datetime.now(UTC),
        )


class ImgwClient:
    """Fetches latest IMGW synop CSV and aggregates only selected coastal stations."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_latest(self) -> ClientResult:
        response = requests.get(
            "https://danepubliczne.imgw.pl/api/data/synop/format/csv",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        frame = pd.read_csv(StringIO(response.text), sep=None, engine="python")
        if "id_stacji" not in frame.columns:
            raise ValueError("IMGW response missing id_stacji column.")

        frame["id_stacji"] = pd.to_numeric(frame["id_stacji"], errors="coerce")
        filtered = frame[frame["id_stacji"].isin(IMGW_STATION_IDS)].copy()
        if filtered.empty:
            raise ValueError("No target IMGW stations found in CSV payload.")

        filtered["data_pomiaru"] = pd.to_datetime(filtered["data_pomiaru"], errors="coerce")
        filtered["godzina_pomiaru"] = pd.to_numeric(filtered["godzina_pomiaru"], errors="coerce")
        filtered = filtered.sort_values(["data_pomiaru", "godzina_pomiaru"])

        for column in ["temperatura", "wilgotnosc_wzgledna", "suma_opadu", "cisnienie"]:
            filtered[column] = pd.to_numeric(filtered[column], errors="coerce")

        latest = filtered.iloc[-1]
        updated_at = pd.to_datetime(latest["data_pomiaru"], errors="coerce")
        if pd.notna(updated_at) and pd.notna(latest["godzina_pomiaru"]):
            updated_at = updated_at + pd.to_timedelta(int(latest["godzina_pomiaru"]), unit="h")

        return ClientResult(
            values={
                "temperature_c": float(filtered["temperatura"].mean()),
                "humidity_percentage": float(filtered["wilgotnosc_wzgledna"].mean()),
                "rainfall_mm": float(filtered["suma_opadu"].mean()),
                "pressure_hpa": float(filtered["cisnienie"].mean()),
            },
            updated_at=updated_at.to_pydatetime() if pd.notna(updated_at) else None,
        )
