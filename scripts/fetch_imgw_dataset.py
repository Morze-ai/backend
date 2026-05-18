#!/usr/bin/env python
"""
Script to fetch meteorological datasets from IMGW (Polish Institute of Meteorology and Water Management).
alongside hydrological datasets where available.

This should be extended to as many stations as possible (alongside the shore), currently we have the following:

Station - ID
- Gdańsk - 12155
- Hel - 12135
- Kołobrzeg - 12100
- Lębork - 12125
- Łeba - 12120
- Świnoujście - 12200
- Ustka - 12115

format:
id_stacji  stacja  data_pomiaru  godzina_pomiaru  temperatura  predkosc_wiatru  kierunek_wiatru  wilgotnosc_wzgledna  suma_opadu  cisnienie

Empty values are marked as "", but we will convert them to NaN for easier processing.

Gives today's data only. For historical data, we need to fetch from archives... `data/raw/[year]`
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

COASTAL_STATION_IDS = [12155, 12135, 12100, 12125, 12120, 12200, 12115]


def fetch_imgw_data(station_id: int | None = None) -> None:
    """
    Fetches the latest meteorological data for a given station ID from IMGW.

    Args:
        station_id (int): The ID of the station to fetch data for.
    """
    url = "https://danepubliczne.imgw.pl/api/data/synop/format/csv"
    response = requests.get(url, timeout=20)

    if response.status_code == 200:
        data = response.text
        df = pd.read_csv(StringIO(data), sep=None, engine="python")
        df.replace("", pd.NA, inplace=True)  # Convert empty strings to NaN

        if "id_stacji" not in df.columns:
            raise ValueError("IMGW payload missing required column: id_stacji")

        df["id_stacji"] = pd.to_numeric(df["id_stacji"], errors="coerce")
        allowed_station_ids = [station_id] if station_id is not None else COASTAL_STATION_IDS
        filtered = df[df["id_stacji"].isin(allowed_station_ids)].copy()

        if filtered.empty:
            raise ValueError("No rows for selected IMGW stations were returned.")

        output_name = (
            "imgw_coastal_latest.csv" if station_id is None else f"{station_id}_latest.csv"
        )
        filtered.to_csv(f"data/raw/{output_name}", index=False)
        print(
            f"Saved {len(filtered)} IMGW records for stations {sorted(set(filtered['id_stacji'].dropna().astype(int).tolist()))}."
        )
    else:
        print(f"Failed to fetch IMGW data. Status code: {response.status_code}")
