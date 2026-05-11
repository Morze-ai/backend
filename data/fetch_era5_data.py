import os
from pathlib import Path

import cdsapi

client = cdsapi.Client(url=os.getenv("CDSAPI_URL"), key=os.getenv("CDSAPI_KEY"))

variables = {
    "pressure": "mean_sea_level_pressure",
    "total_precipitation": "total_precipitation",
    "sea_surface_temperature": "sea_surface_temperature",
}

for year in range(2021, 2026):
    for folder_name, variable_name in variables.items():
        folder = Path(f"raw/{year}/{folder_name}")
        folder.mkdir(parents=True, exist_ok=True)

        request = {
            "product_type": ["reanalysis"],
            "variable": [variable_name],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [54.5, 18.3, 54.2, 18.9],
        }

        output_path = str(folder / "data.nc")

        print(f"Downloading {variable_name} for {year}...")

        client.retrieve("reanalysis-era5-single-levels", request).download(output_path)

        print(f"Saved to {output_path}")
