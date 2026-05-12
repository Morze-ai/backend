"""Simple validators for ERA5-derived DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def check_temporal_coverage(df: pd.DataFrame, freq: str = "H") -> dict:
    if df.empty:
        return {"status": "empty", "missing_hours": None}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    missing = full.difference(cast(pd.DatetimeIndex, df.index))
    return {"status": "ok", "total_hours": len(full), "missing_hours": len(missing)}


def missing_value_report(df: pd.DataFrame) -> dict:
    report = {}
    for col in df.columns:
        if col == "timestamp":
            continue
        pct = df[col].isna().mean() * 100.0
        report[col] = float(pct)
    return report


def save_report(path: Path, coverage: dict, missing_report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ERA5 validation report\n")
        f.write("Coverage:\n")
        for k, v in coverage.items():
            f.write(f"  {k}: {v}\n")
        f.write("Missing value percentages:\n")
        for k, v in missing_report.items():
            f.write(f"  {k}: {v:.2f}%\n")
