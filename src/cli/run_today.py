"""Run short-term forecast predictions using historical data + Open-Meteo forecasts.

This script builds full engineered features for each target horizon by
concatenating recent history from the configured raw CSV with Open-Meteo hourly
forecasts up to the prediction timestamp, then applies the saved preprocessor
and runs model inference.

It also handles the common checkpoint vs config feature mismatch by aligning
`config.data.feature_columns` with the preprocessor artifact when needed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import requests
import yaml

from src.cli import build_experiment, load_raw_frame
from src.data.feature_engineering import (
    drop_initial_lag_rows,
    engineer_features,
    generate_lag_features,
    generate_rolling_features,
    generate_seasonal_features,
)
from src.data.preprocessing import PreprocessorStats, apply_preprocessor
from src.models.calibration import apply_temperature_scaling, load_temperature_scaling
from src.utils.io import read_json


def fetch_open_meteo(lat: float, lon: float, past_days: int = 10, forecast_days: int = 7):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,"
            "wind_speed_10m,wind_direction_10m"
        ),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15.0)
    r.raise_for_status()
    return r.json()


def hourly_payload_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    df = pd.DataFrame(
        {
            "timestamp": times,
            "temperature_c": hourly.get("temperature_2m", []),
            "humidity_percentage": hourly.get("relative_humidity_2m", []),
            "rainfall_mm": hourly.get("precipitation", []),
            "pressure_hpa": hourly.get("pressure_msl", []),
            "wind_speed": hourly.get("wind_speed_10m", []),
            "wind_direction": hourly.get("wind_direction_10m", []),
        }
    )
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def prepare_combined_frame(
    raw_frame: pd.DataFrame, forecast_df: pd.DataFrame, target_ts: pd.Timestamp, history_hours: int
) -> pd.DataFrame:
    # Ensure timestamp dtype
    raw = raw_frame.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw = raw.sort_values("timestamp")

    # History window: take last `history_hours` from raw
    window_start = target_ts - pd.Timedelta(hours=history_hours)
    history = raw.loc[raw["timestamp"] >= window_start].copy()

    # Keep forecast rows up to and including target timestamp
    forecast_rows = forecast_df.loc[forecast_df["timestamp"] <= target_ts].copy()

    combined = pd.concat([history, forecast_rows], ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    # Fill missing weather columns from forecast where available
    for col in [
        "temperature_c",
        "humidity_percentage",
        "rainfall_mm",
        "pressure_hpa",
        "wind_speed",
        "wind_direction",
    ]:
        if col not in combined.columns:
            combined[col] = np.nan

    return combined


def build_features_for_target(combined: pd.DataFrame, config) -> pd.DataFrame:
    frame = combined.copy()
    # The project's engineer_features expects certain column names like rainfall_mm etc.
    frame = engineer_features(frame)

    # Generate lag features if configured
    if config.feature_engineering.generate_lag_features:
        lag_hours = config.feature_engineering.lag_hours
        # Map available weather columns to lag generator expected mapping
        weather_columns = {
            "rainfall_mm": lag_hours,
            "temperature_c": lag_hours,
            "pressure_hpa": lag_hours,
        }
        available = {k: v for k, v in weather_columns.items() if k in frame.columns}
        if available:
            frame = generate_lag_features(
                frame, timestamp_column="timestamp", lag_columns=available, use_cuda=False
            )
            frame = drop_initial_lag_rows(
                frame, max_lag_hours=max(available.values()), timestamp_column="timestamp"
            )

    # Rolling features
    if config.feature_engineering.generate_rolling_features:
        windows = config.feature_engineering.rolling_windows
        rolling_base_columns = [
            c
            for c in ["rainfall_mm", "temperature_c", "pressure_hpa", "humidity_percentage"]
            if c in frame.columns
        ]
        if rolling_base_columns:
            frame = generate_rolling_features(
                frame,
                window_hours=windows,
                agg_functions=["mean", "max", "min"],
                columns_to_aggregate=rolling_base_columns,
                timestamp_column="timestamp",
            )

    # Seasonal features
    if config.feature_engineering.generate_seasonal_features:
        frame = generate_seasonal_features(frame, timestamp_column="timestamp")

    return frame


def fill_feature_fallbacks(frame: pd.DataFrame, config) -> pd.DataFrame:
    df = frame.copy()
    # Ensure basic weather columns are forward/backfilled from available data
    base_cols = [
        "rainfall_mm",
        "temperature_c",
        "pressure_hpa",
        "humidity_percentage",
        "wind_speed",
        "wind_direction",
    ]
    for col in base_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].fillna(0.0)

    # Compute wind_u / wind_v from speed+direction if missing or NaN
    if (
        ("wind_u" not in df.columns or df["wind_u"].isna().all())
        and "wind_speed" in df.columns
        and "wind_direction" in df.columns
    ):
        radians = np.deg2rad(df["wind_direction"].fillna(0.0).to_numpy(dtype=float, copy=True))
        ws = df["wind_speed"].fillna(0.0).to_numpy(dtype=float, copy=True)
        df["wind_u"] = (ws * np.sin(radians)).tolist()
        df["wind_v"] = (ws * np.cos(radians)).tolist()

    # Fill lag features with most recent base value if missing
    lag_cols = [c for c in df.columns if "_lag_" in c]
    for col in lag_cols:
        # base name is part before _lag_
        base = col.split("_lag_")[0]
        if base in df.columns:
            base_series = cast(pd.Series, df[base]).ffill().bfill()
            last_val = float(base_series.iloc[-1]) if not base_series.empty else 0.0
        else:
            last_val = 0.0
        df[col] = df[col].fillna(last_val)

    # Rolling aggregates: fill with base mean if present
    for func in ["_mean_", "_max_", "_min_"]:
        rc = [c for c in df.columns if func in c]
        for col in rc:
            # attempt to infer base prefix (e.g., rainfall_mm_mean_3h -> rainfall_mm)
            parts = col.split(func)
            base = parts[0]
            if base in df.columns:
                base_mean = float(df[base].mean()) if not df[base].dropna().empty else 0.0
            else:
                base_mean = 0.0
            df[col] = df[col].fillna(base_mean)

    # temp_delta_24h: try compute from available temperatures
    if "temp_delta_24h" in df.columns:
        try:
            last_ts = df["timestamp"].max()
            last_temp = (
                cast(pd.Series, df.loc[df["timestamp"] == last_ts, "temperature_c"]).ffill().bfill()
            )
            if not last_temp.empty:
                last_temp_val = float(last_temp.iloc[0])
            else:
                last_temp_val = float(
                    cast(pd.Series, df["temperature_c"]).ffill().bfill().iloc[-1]
                    if not cast(pd.Series, df["temperature_c"]).ffill().bfill().empty
                    else 0.0
                )

            ts_24 = last_ts - pd.Timedelta(hours=24)
            prev = (
                cast(pd.Series, df.loc[df["timestamp"] <= ts_24, "temperature_c"]).ffill().bfill()
            )
            prev_val = float(prev.iloc[-1]) if not prev.empty else last_temp_val
            df["temp_delta_24h"] = df["temp_delta_24h"].fillna(last_temp_val - prev_val)
        except Exception:
            df["temp_delta_24h"] = df["temp_delta_24h"].fillna(0.0)

    # Known domain features defaults
    if "thaw_flag" in df.columns:
        df["thaw_flag"] = df["thaw_flag"].fillna(0)
    if "soil_saturation_index" in df.columns:
        df["soil_saturation_index"] = df["soil_saturation_index"].fillna(0.0)

    # Ensure no Inf or NaN remain
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    for c in numeric_cols:
        arr = pd.to_numeric(df[c], errors="coerce")
        arr = arr.fillna(0.0)
        arr[~np.isfinite(arr.to_numpy(dtype=float, copy=True))] = 0.0
        df[c] = arr

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "config_path",
        nargs="?",
        help="Path to model config YAML (optional). If omitted all configs in configs/ will be run.",
    )
    p.add_argument("--lat", type=float, default=54.352, help="Latitude")
    p.add_argument("--lon", type=float, default=18.6466, help="Longitude")
    p.add_argument(
        "--safety",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable safety checks for NaN/Inf in features (1) or disable (0)",
    )
    p.add_argument(
        "--force",
        type=int,
        default=0,
        choices=[0, 1],
        help="Force prediction even if safety checks fail",
    )
    args = p.parse_args()

    # If no config_path provided, run for all YAML files under configs/
    config_paths = []
    if args.config_path:
        config_paths = [args.config_path]
    else:
        config_dir = Path("configs")
        candidates = sorted(config_dir.glob("*.yaml"))
        config_paths = []
        for p in candidates:
            try:
                payload = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and ("data" in payload or "model" in payload):
                    config_paths.append(str(p))
            except Exception:
                continue

    # Fetch forecast payload once and reuse for all runs
    today = date.today()
    payload = fetch_open_meteo(args.lat, args.lon, past_days=10, forecast_days=7)
    forecast_df = hourly_payload_to_df(payload)

    for cfg in config_paths:
        try:
            _process_single_config(
                cfg,
                args.lat,
                args.lon,
                today,
                payload,
                forecast_df,
                safety=bool(args.safety),
                force=bool(args.force),
            )
        except Exception as e:
            print(f"Error processing {cfg}: {e}")


def _process_single_config(
    config_path: str,
    lat: float,
    lon: float,
    today: date,
    payload: dict,
    forecast_df: pd.DataFrame,
    *,
    safety: bool = True,
    force: bool = False,
) -> None:
    """Process a single YAML config path. Extracted for multi-config runs."""
    # Build experiment + load project config
    config, exp = build_experiment(config_path)

    # Align config feature columns to preprocessor artifact if necessary
    try:
        preproc = read_json(config.paths.preprocessor_artifact)
        preproc_features = list(preproc.get("features", {}).keys())
        if preproc_features and preproc_features != list(config.data.feature_columns):
            print("⚠️  Aligning config.feature_columns to preprocessor artifact (runtime override).")
            config.data.feature_columns = preproc_features
    except Exception:
        # Preprocessor not found or invalid; we'll still attempt to build features
        preproc = None

    # Determine history window
    history_hours = 0
    try:
        if config.feature_engineering.generate_lag_features:
            history_hours = max(history_hours, int(config.feature_engineering.lag_hours))
        if config.feature_engineering.generate_rolling_features:
            history_hours = max(
                history_hours, max(list(config.feature_engineering.rolling_windows))
            )
    except Exception:
        history_hours = 72

    horizons = {
        "+1d": today + timedelta(days=1),
        "+3d": today + timedelta(days=3),
        "+7d": today + timedelta(days=7),
    }

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_reference": config_path,
        "forecasts": {},
        "predictions": {},
    }

    raw_frame = load_raw_frame(config)

    for label, target_date in horizons.items():
        # select an hour for the target day (prefer 12:00 UTC if available)
        day_start = pd.Timestamp(target_date.isoformat(), tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)
        candidates = forecast_df.loc[
            (forecast_df["timestamp"] >= day_start) & (forecast_df["timestamp"] < day_end)
        ].copy()
        if candidates.empty:
            results["forecasts"][label] = {}
            results["predictions"][label] = {"error": "no forecast data for day"}
            continue

        # pick 12:00 if available else median
        target_ts = None
        mid = day_start + pd.Timedelta(hours=12)
        if mid in candidates["timestamp"].values:
            target_ts = mid
        else:
            target_ts = candidates.iloc[len(candidates) // 2]["timestamp"]

        # build combined frame with recent history + forecast hours up to target
        combined = prepare_combined_frame(raw_frame, forecast_df, target_ts, history_hours)
        features_frame = build_features_for_target(combined, config)
        # Fill fallback values to avoid NaNs from insufficient history
        features_frame = fill_feature_fallbacks(features_frame, config)
        # Defragment the DataFrame after many column insertions to avoid
        # pandas PerformanceWarning about fragmentation caused by repeated
        # column assignments. A single deep copy consolidates memory.
        features_frame = features_frame.copy()

        # select the row corresponding to target timestamp (closest)
        features_frame["_time_diff"] = (features_frame["timestamp"] - target_ts).abs()
        row = (
            features_frame.sort_values("_time_diff").iloc[0:1].drop(columns=["_time_diff"])
            if not features_frame.empty
            else None
        )

        results["forecasts"][label] = {"target_timestamp": str(target_ts)}

        if row is None or row.empty:
            results["predictions"][label] = {"error": "unable to construct feature row"}
            continue

        # Apply preprocessor stats if available
        if preproc:
            raw_stats = cast(PreprocessorStats, preproc["features"])
            transformed = apply_preprocessor(
                frame=row[config.data.feature_columns],
                feature_columns=config.data.feature_columns,
                stats=raw_stats,
            )
        else:
            transformed = row[config.data.feature_columns]

        # Safety checks: ensure no NaN/Inf in transformed input
        safety_enabled = safety
        force = force
        has_nan = transformed.isna().any(axis=None)
        has_inf = (~np.isfinite(transformed.to_numpy(dtype=float, copy=True))).any()
        if safety_enabled and (has_nan or has_inf) and not force:
            results["predictions"][label] = {
                "error": "safety_check_failed",
                "details": {
                    "has_nan": bool(has_nan),
                    "has_inf": bool(has_inf),
                    "message": "Transformed feature row contains NaN/Inf. Use --force to override.",
                },
            }
            continue

        # sanitize input: replace NaN/Inf with 0.0
        transformed = transformed.fillna(0.0)
        arr = transformed.to_numpy(dtype=float, copy=True)
        arr[~np.isfinite(arr)] = 0.0
        transformed = pd.DataFrame(arr, columns=config.data.feature_columns)

        # Run model prediction
        try:
            model = exp.build_model()
            exp.load_checkpoint(model)
            temperature = load_temperature_scaling(exp._calibration_path())
            preds, probs = apply_temperature_scaling(
                model=model,
                frame=transformed,
                feature_columns=config.data.feature_columns,
                class_names=config.data.class_names,
                task_type=exp._task_type(),
                temperature=temperature,
            )

            # Post-process probabilities: replace NaN/Inf, normalize, fallback to one-hot if needed
            proc_probs = []
            for i, p_row in enumerate(probs):
                p_arr = np.array(p_row, dtype=float)
                # replace non-finite
                p_arr[~np.isfinite(p_arr)] = 0.0
                s = float(np.sum(p_arr))
                if s > 0:
                    p_arr = p_arr / s
                else:
                    # fallback: one-hot on predicted class
                    pred = preds[i]
                    onehot = np.zeros(len(config.data.class_names), dtype=float)
                    try:
                        idx = config.data.class_names.index(pred)
                        onehot[idx] = 1.0
                    except Exception:
                        onehot[0] = 1.0
                    p_arr = onehot

                proc_probs.append(p_arr.tolist())

            results["predictions"][label] = {
                "predicted_class": preds[0],
                "probabilities": proc_probs[0],
            }
        except Exception as e:
            results["predictions"][label] = {"error": str(e)}

    out_dir = Path("reports") / Path(config_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"run_today_{today.isoformat()}.json"
    out_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved run-today output to {out_file}")


if __name__ == "__main__":
    main()
