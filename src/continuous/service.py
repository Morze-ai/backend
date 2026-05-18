"""Continuous inference orchestration service for live flood-risk evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from src.cli import load_project_config
from src.continuous.clients import ClientResult, ImgwClient, OpenMeteoClient, StormglassClient
from src.continuous.schemas import (
    ContinuousEvaluationResult,
    HistoricalComparison,
    HorizonEvaluation,
    OnsetWindow,
    ReportArtifacts,
    SourceStatus,
)
from src.data.feature_engineering import (
    drop_initial_lag_rows,
    engineer_features,
    generate_lag_features,
    generate_rolling_features,
    generate_seasonal_features,
)
from src.data.preprocessing import PreprocessorStats, apply_preprocessor
from src.events.attribution import attribute_event_type, compute_historical_confidence
from src.events.evaluator import build_detector_output_frame
from src.events.schemas import EventType
from src.experiments.registry import ExperimentFactory
from src.explain.feature_importance import rank_features, top_k_features
from src.explain.shap_explainer import ShapAnalyzer
from src.models.calibration import apply_temperature_scaling, load_temperature_scaling
from src.utils.io import ensure_parent, read_json


class OpenMeteoFetcher(Protocol):
    """Protocol for Open-Meteo compatible fetchers."""

    def fetch_latest(self, latitude: float, longitude: float) -> ClientResult: ...

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        past_days: int = ...,
        forecast_days: int = ...,
    ) -> pd.DataFrame: ...


class StormglassFetcher(Protocol):
    """Protocol for Stormglass compatible fetchers."""

    def fetch_latest(self, latitude: float, longitude: float) -> ClientResult: ...


class ImgwFetcher(Protocol):
    """Protocol for IMGW compatible fetchers."""

    def fetch_latest(self) -> ClientResult: ...


@dataclass(slots=True)
class ServiceOptions:
    """Runtime options for live inference orchestration."""

    latitude: float = 54.352
    longitude: float = 18.6466
    history_rows: int = 240


class _SnapshotResult(TypedDict):
    """Typed return from _evaluate_current_snapshot."""

    predicted_class: str
    risk_level_score: float
    confidence_score: float
    onset_window: OnsetWindow
    dominant_factor: str
    historical: HistoricalComparison
    event_type: str
    event_message: str
    summary: str


class ContinuousEvaluationService:
    """Builds a continuous risk result from live and historical data."""

    def __init__(
        self,
        config_path: str,
        *,
        options: ServiceOptions | None = None,
        open_meteo_client: OpenMeteoFetcher | None = None,
        stormglass_client: StormglassFetcher | None = None,
        imgw_client: ImgwFetcher | None = None,
    ) -> None:
        self.config_path = config_path
        self.options = options or ServiceOptions()
        load_dotenv()
        # Load config first so we can align features to any saved preprocessor artifact
        self.config = load_project_config(config_path)

        # Align config.feature_columns to the saved preprocessor artifact if available.
        self._preprocessor_stats: PreprocessorStats | None = None
        try:
            preproc_payload = read_json(self.config.paths.preprocessor_artifact)
            preproc_features = list(preproc_payload.get("features", {}).keys())
            if preproc_features and preproc_features != list(self.config.data.feature_columns):
                print(
                    "⚠️  Aligning config.feature_columns to preprocessor artifact (runtime override)."
                )
                self.config.data.feature_columns = preproc_features
            raw_stats = preproc_payload.get("features")
            if isinstance(raw_stats, dict):
                self._preprocessor_stats = raw_stats
        except Exception:
            self._preprocessor_stats = None

        # Build experiment from (possibly adjusted) config
        self.experiment = ExperimentFactory.build(self.config.model.name, self.config)
        self.experiment.config_path = Path(config_path).resolve()
        stormglass_api_key = self._resolve_stormglass_key()
        self.open_meteo_client = open_meteo_client or OpenMeteoClient()
        self.stormglass_client = stormglass_client or StormglassClient(stormglass_api_key)
        self.imgw_client = imgw_client or ImgwClient()

    def evaluate(self, persist: bool = True) -> ContinuousEvaluationResult:
        """Run one complete continuous evaluation."""

        history = self._load_history()
        source_status: list[SourceStatus] = []
        live_values: dict[str, float] = {}

        live_values, source_status = self._merge_live_sources()

        history_with_live = self._append_live_row(history, live_values)
        current_snapshot = self._evaluate_current_snapshot(history_with_live, source_status)

        horizon_forecasts: dict[str, HorizonEvaluation] = {}
        warnings: list[str] = [
            f"source_unavailable:{item.name}" for item in source_status if not item.ok
        ]
        forecast_frame: pd.DataFrame | None = None
        try:
            forecast_frame = self.open_meteo_client.fetch_forecast(
                latitude=self.options.latitude,
                longitude=self.options.longitude,
                forecast_days=14,
            )
        except Exception as exc:
            warnings.append(f"open_meteo_forecast_unavailable: {exc}")
            source_status.append(
                SourceStatus(name="open_meteo_forecast", ok=False, detail=str(exc), updated_at=None)
            )

        if forecast_frame is None or forecast_frame.empty:
            warnings.append("open_meteo_forecast_empty")
        else:
            horizon_forecasts = self._evaluate_horizons(history_with_live, forecast_frame)

        report = self._write_forecast_report(horizon_forecasts, warnings)

        result = ContinuousEvaluationResult(
            model_reference=self.config_path,
            predicted_risk_level=current_snapshot["predicted_class"],
            risk_level_score=current_snapshot["risk_level_score"],
            confidence_score=current_snapshot["confidence_score"],
            expected_onset_window=current_snapshot["onset_window"],
            dominant_contributing_factor=current_snapshot["dominant_factor"],
            historical_comparison=current_snapshot["historical"],
            source_status=source_status,
            event_type=current_snapshot["event_type"],
            event_message=current_snapshot["event_message"],
            summary=current_snapshot["summary"],
            horizon_forecasts=horizon_forecasts,
            report=report,
            warnings=warnings,
        )

        if persist:
            self._persist_result(result)

        return result

    def get_latest(self) -> ContinuousEvaluationResult | None:
        """Read last persisted result if available."""

        path = self._continuous_latest_path()
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ContinuousEvaluationResult.model_validate(payload)

    def get_history(self, limit: int = 50) -> list[ContinuousEvaluationResult]:
        """Read recent persisted history entries."""

        path = self._continuous_history_path()
        if not path.exists():
            return []

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = lines[-limit:]
        return [ContinuousEvaluationResult.model_validate_json(line) for line in selected]

    def _load_history(self) -> pd.DataFrame:
        frame = pd.read_csv(self.config.paths.raw_csv)
        if "timestamp" not in frame.columns:
            raise ValueError("Configured raw CSV must include a timestamp column.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        return frame.tail(self.options.history_rows).reset_index(drop=True)

    def _append_live_row(
        self, history: pd.DataFrame, live_values: dict[str, float]
    ) -> pd.DataFrame:
        if history.empty:
            return history
        base_row = history.iloc[-1].to_dict()
        base_row.update(live_values)
        base_row["timestamp"] = datetime.now(UTC).isoformat()
        return pd.concat([history, pd.DataFrame([base_row])], ignore_index=True)

    def _evaluate_current_snapshot(
        self,
        merged: pd.DataFrame,
        source_status: list[SourceStatus],
    ) -> _SnapshotResult:
        merged = engineer_features(merged)
        merged = generate_seasonal_features(merged, timestamp_column="timestamp")

        merged = merged.ffill().bfill().fillna(0.0)

        if ("wind_u" not in merged.columns or merged["wind_u"].isna().all()) and (
            "wind_speed" in merged.columns and "wind_direction" in merged.columns
        ):
            radians = np.deg2rad(
                merged["wind_direction"].fillna(0.0).to_numpy(dtype=float, copy=True)
            )
            ws = merged["wind_speed"].fillna(0.0).to_numpy(dtype=float, copy=True)
            merged["wind_u"] = (ws * np.sin(radians)).tolist()
            merged["wind_v"] = (ws * np.cos(radians)).tolist()

        candidate_row = merged.iloc[-1].to_dict()
        required_features = self.config.data.feature_columns
        raw_values: dict[str, float] = {
            feature: float(candidate_row.get(feature, 0.0) or 0.0) for feature in required_features
        }

        predicted_class, probabilities = self.experiment.predict_one(raw_values)
        risk_score = float(max(probabilities))

        detection = self._detect_event(merged)

        event_type = str(detection.get("event_type", "none")) if detection is not None else "none"
        detection_confidence = (
            float(detection.get("detection_confidence") or 0.0) if detection is not None else 0.0
        )
        detection_severity = (
            float(detection.get("detection_severity") or 0.0) if detection is not None else 0.0
        )
        event_message = str(detection.get("detection_message", "")) if detection is not None else ""

        onset_window = self._estimate_onset_window(event_type, risk_score, detection_confidence)
        dominant_factor = self._dominant_factor(detection, event_type)
        historical = self._historical_comparison(event_type, merged)

        confidence_score = min(1.0, (risk_score * 0.6) + (historical.historical_confidence * 0.4))
        if not all(item.ok for item in source_status):
            confidence_score *= 0.85

        summary = (
            f"Risk={predicted_class} ({risk_score:.2f}), event={event_type}, "
            f"factor={dominant_factor}, confidence={confidence_score:.2f}."
        )

        return _SnapshotResult(
            predicted_class=predicted_class,
            risk_level_score=max(risk_score, detection_severity),
            confidence_score=confidence_score,
            onset_window=onset_window,
            dominant_factor=dominant_factor,
            historical=historical,
            event_type=event_type,
            event_message=event_message,
            summary=summary,
        )

    def _evaluate_horizons(
        self,
        history: pd.DataFrame,
        forecast_frame: pd.DataFrame,
    ) -> dict[str, HorizonEvaluation]:
        horizons = {
            "+1d": 1,
            "+3d": 3,
            "+7d": 7,
        }

        history_hours = self._required_history_hours()
        model = self.experiment.build_model()
        self.experiment.load_checkpoint(model)
        temperature = load_temperature_scaling(self.experiment._calibration_path())
        task_type = self.experiment._task_type()

        results: dict[str, HorizonEvaluation] = {}
        today = pd.Timestamp(datetime.now(UTC))

        for label, offset_days in horizons.items():
            target_date = (today + pd.Timedelta(days=offset_days)).date()
            day_start = pd.Timestamp(target_date.isoformat(), tz="UTC")
            day_end = day_start + pd.Timedelta(days=1)
            candidates = forecast_frame.loc[
                (forecast_frame["timestamp"] >= day_start) & (forecast_frame["timestamp"] < day_end)
            ].copy()

            horizon = HorizonEvaluation(horizon=label)
            warnings: list[str] = []

            if candidates.empty:
                horizon.warnings.append("no_forecast_for_horizon")
                results[label] = horizon
                continue

            mid = day_start + pd.Timedelta(hours=12)
            if mid in candidates["timestamp"].values:
                target_ts = mid
            else:
                target_ts = candidates.iloc[len(candidates) // 2]["timestamp"]

            combined = self._prepare_combined_frame(
                history, forecast_frame, target_ts, history_hours
            )
            features_frame = self._build_features_for_target(combined)
            features_frame = self._fill_feature_fallbacks(features_frame)
            features_frame = features_frame.copy()

            missing_features = [
                feature
                for feature in self.config.data.feature_columns
                if feature not in features_frame.columns
            ]
            if missing_features:
                warnings.append("missing_features:" + ",".join(missing_features[:10]))

            features_frame["_time_diff"] = (features_frame["timestamp"] - target_ts).abs()
            row = (
                features_frame.sort_values("_time_diff").iloc[0:1].drop(columns=["_time_diff"])
                if not features_frame.empty
                else None
            )

            if row is None or row.empty:
                horizon.warnings.append("unable_to_build_feature_row")
                results[label] = horizon
                continue

            transformed = self._apply_preprocessor(row)
            transformed = self._sanitize_numeric_frame(transformed)

            predictions, probabilities = apply_temperature_scaling(
                model=model,
                frame=transformed,
                feature_columns=self.config.data.feature_columns,
                class_names=self.config.data.class_names,
                task_type=task_type,
                temperature=temperature,
            )

            predicted_class = predictions[0]
            risk_score = float(max(probabilities[0]))

            detection = self._detect_event(features_frame)
            event_type = (
                str(detection.get("event_type", "none")) if detection is not None else "none"
            )
            detection_confidence = (
                float(detection.get("detection_confidence") or 0.0)
                if detection is not None
                else 0.0
            )
            detection_severity = (
                float(detection.get("detection_severity") or 0.0) if detection is not None else 0.0
            )
            event_message = (
                str(detection.get("detection_message", "")) if detection is not None else ""
            )

            onset_window = self._estimate_onset_window(event_type, risk_score, detection_confidence)
            historical = self._historical_comparison(event_type, features_frame)
            confidence_score = min(
                1.0, (risk_score * 0.6) + (historical.historical_confidence * 0.4)
            )
            if confidence_score < 0.5:
                warnings.append("low_confidence_forecast")

            top_features = self._compute_top_shap_features(
                model=model,
                features_frame=features_frame,
                target_row=transformed,
            )

            if not top_features:
                warnings.append("shap_unavailable")

            main_cause = None
            if top_features:
                main_cause = attribute_event_type(top_features).value

            dominant_factor = self._dominant_factor(detection, event_type)
            if not dominant_factor or dominant_factor == "none":
                dominant_factor = main_cause or event_type

            horizon.target_timestamp = target_ts.to_pydatetime()
            if row is not None and not row.empty:
                horizon.forecasted_temperature_c = (
                    float(row["temperature_c"].iloc[0]) if "temperature_c" in row else None
                )
                horizon.forecasted_rainfall_mm = (
                    float(row["rainfall_mm"].iloc[0]) if "rainfall_mm" in row else None
                )
                horizon.forecasted_pressure_hpa = (
                    float(row["pressure_hpa"].iloc[0]) if "pressure_hpa" in row else None
                )
                horizon.forecasted_wind_speed_ms = (
                    float(row["wind_speed"].iloc[0]) if "wind_speed" in row else None
                )
            horizon.predicted_risk_level = predicted_class
            horizon.risk_level_score = max(risk_score, detection_severity)
            horizon.confidence_score = confidence_score
            horizon.expected_onset_window = onset_window
            horizon.dominant_contributing_factor = dominant_factor
            horizon.main_cause = main_cause
            horizon.historical_comparison = historical
            horizon.event_type = event_type
            horizon.event_message = event_message
            horizon.top_features = top_features
            horizon.warnings.extend(warnings)

            results[label] = horizon

        return results

    def _required_history_hours(self) -> int:
        history_hours = 0
        try:
            if self.config.feature_engineering.generate_lag_features:
                history_hours = max(history_hours, int(self.config.feature_engineering.lag_hours))
            if self.config.feature_engineering.generate_rolling_features:
                history_hours = max(
                    history_hours, max(list(self.config.feature_engineering.rolling_windows))
                )
        except Exception:
            history_hours = 72
        return history_hours

    def _prepare_combined_frame(
        self,
        raw_frame: pd.DataFrame,
        forecast_frame: pd.DataFrame,
        target_ts: pd.Timestamp,
        history_hours: int,
    ) -> pd.DataFrame:
        raw = raw_frame.copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        raw = raw.sort_values("timestamp")

        window_start = target_ts - pd.Timedelta(hours=history_hours)
        history = raw.loc[raw["timestamp"] >= window_start].copy()

        forecast_rows = forecast_frame.loc[forecast_frame["timestamp"] <= target_ts].copy()

        combined = pd.concat([history, forecast_rows], ignore_index=True, sort=False)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        for col in [
            "temperature_c",
            "humidity_percentage",
            "rainfall_mm",
            "pressure_hpa",
            "wind_speed",
            "wind_direction",
            "wind_u",
            "wind_v",
        ]:
            if col not in combined.columns:
                combined[col] = np.nan

        return combined

    def _build_features_for_target(self, combined: pd.DataFrame) -> pd.DataFrame:
        frame = combined.copy()
        frame = engineer_features(frame)

        if self.config.feature_engineering.generate_lag_features:
            lag_hours = self.config.feature_engineering.lag_hours
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

        if self.config.feature_engineering.generate_rolling_features:
            windows = self.config.feature_engineering.rolling_windows
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

        if self.config.feature_engineering.generate_seasonal_features:
            frame = generate_seasonal_features(frame, timestamp_column="timestamp")

        return frame

    def _fill_feature_fallbacks(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        base_cols = [
            "rainfall_mm",
            "temperature_c",
            "pressure_hpa",
            "humidity_percentage",
            "wind_speed",
            "wind_direction",
            "wind_u",
            "wind_v",
        ]
        for col in base_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0.0)

        if (
            ("wind_u" not in df.columns or df["wind_u"].isna().all())
            and "wind_speed" in df.columns
            and "wind_direction" in df.columns
        ):
            radians = np.deg2rad(df["wind_direction"].fillna(0.0).to_numpy(dtype=float, copy=True))
            ws = df["wind_speed"].fillna(0.0).to_numpy(dtype=float, copy=True)
            df["wind_u"] = (ws * np.sin(radians)).tolist()
            df["wind_v"] = (ws * np.cos(radians)).tolist()

        lag_cols = [c for c in df.columns if "_lag_" in c]
        for col in lag_cols:
            base = col.split("_lag_")[0]
            if base in df.columns:
                base_series = df[base].ffill().bfill()
                last_val = float(base_series.iloc[-1]) if not base_series.empty else 0.0
            else:
                last_val = 0.0
            df[col] = df[col].fillna(last_val)

        for func in ["_mean_", "_max_", "_min_"]:
            rc = [c for c in df.columns if func in c]
            for col in rc:
                parts = col.split(func)
                base = parts[0]
                if base in df.columns:
                    base_mean = float(df[base].mean()) if not df[base].dropna().empty else 0.0
                else:
                    base_mean = 0.0
                df[col] = df[col].fillna(base_mean)

        if "temp_delta_24h" in df.columns:
            try:
                last_ts = df["timestamp"].max()
                last_temp = (
                    cast(pd.Series, df.loc[df["timestamp"] == last_ts, "temperature_c"])
                    .ffill()
                    .bfill()
                )
                if not last_temp.empty:
                    last_temp_val = float(last_temp.iloc[0])
                else:
                    temp_series = df["temperature_c"].ffill().bfill()
                    last_temp_val = float(temp_series.iloc[-1]) if not temp_series.empty else 0.0

                ts_24 = last_ts - pd.Timedelta(hours=24)
                prev = (
                    cast(pd.Series, df.loc[df["timestamp"] <= ts_24, "temperature_c"])
                    .ffill()
                    .bfill()
                )
                prev_val = float(prev.iloc[-1]) if not prev.empty else last_temp_val
                df["temp_delta_24h"] = df["temp_delta_24h"].fillna(last_temp_val - prev_val)
            except Exception:
                df["temp_delta_24h"] = df["temp_delta_24h"].fillna(0.0)

        if "thaw_flag" in df.columns:
            df["thaw_flag"] = df["thaw_flag"].fillna(0)
        if "soil_saturation_index" in df.columns:
            df["soil_saturation_index"] = df["soil_saturation_index"].fillna(0.0)

        numeric_cols = df.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            arr[~np.isfinite(arr.to_numpy(dtype=float, copy=True))] = 0.0
            df[col] = arr

        return df

    def _apply_preprocessor(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = self._ensure_feature_columns(frame)
        if self._preprocessor_stats is None:
            return aligned[self.config.data.feature_columns].copy()
        return apply_preprocessor(
            frame=aligned[self.config.data.feature_columns],
            feature_columns=self.config.data.feature_columns,
            stats=self._preprocessor_stats,
        )

    def _ensure_feature_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        for feature in self.config.data.feature_columns:
            if feature not in aligned.columns:
                aligned[feature] = 0.0
        return aligned

    def _sanitize_numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        sanitized = frame.copy()
        for column in sanitized.columns:
            values = pd.to_numeric(sanitized[column], errors="coerce").fillna(0.0)
            values[~np.isfinite(values.to_numpy(dtype=float, copy=True))] = 0.0
            sanitized[column] = values
        return sanitized

    def _compute_top_shap_features(
        self,
        model: object,
        features_frame: pd.DataFrame,
        target_row: pd.DataFrame,
    ) -> list[str]:
        feature_columns = self.config.data.feature_columns
        if not feature_columns or target_row.empty:
            return []

        try:
            background_frame = self._ensure_feature_columns(features_frame)
            background_frame = background_frame[feature_columns].copy()
            if background_frame.empty:
                return []
            background_frame = self._apply_preprocessor(background_frame)
            background_frame = self._sanitize_numeric_frame(background_frame)
            sample_size = min(64, len(background_frame))
            background = background_frame.tail(sample_size).to_numpy(dtype=float, copy=True)

            analyzer = ShapAnalyzer(model=model, background_data=background)
            shap_values = analyzer.compute_shap_values(target_row.to_numpy(dtype=float, copy=True))
            if shap_values.ndim == 3:
                shap_values = np.mean(np.abs(shap_values), axis=2)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

            importance = rank_features(shap_values, feature_columns)
            top_df = top_k_features(importance, k=5)
            return top_df["feature"].tolist()
        except Exception:
            return []

    def _detect_event(self, merged: pd.DataFrame) -> pd.Series | None:
        detector_columns = [
            "timestamp",
            "rainfall_mm",
            "temperature_c",
            "water_level_m",
            "pressure_hpa",
            "humidity_percentage",
            "rain_24h_sum",
            "rain_1h_sum",
            "rain_3h_sum",
            "rain_6h_sum",
            "rain_12h_sum",
            "rain_24h_sum",
            "temp_mean",
            "temp_delta_24h",
        ]
        available_detector_columns = [c for c in detector_columns if c in merged.columns]
        detection_frame = build_detector_output_frame(merged[available_detector_columns])
        return detection_frame.iloc[-1] if not detection_frame.empty else None

    def _merge_live_sources(self) -> tuple[dict[str, float], list[SourceStatus]]:
        values: dict[str, float] = {}
        statuses: list[SourceStatus] = []

        for name, fetcher in [
            (
                "open_meteo",
                lambda: self.open_meteo_client.fetch_latest(
                    latitude=self.options.latitude,
                    longitude=self.options.longitude,
                ),
            ),
            (
                "stormglass",
                lambda: self.stormglass_client.fetch_latest(
                    latitude=self.options.latitude,
                    longitude=self.options.longitude,
                ),
            ),
            ("imgw", self.imgw_client.fetch_latest),
        ]:
            try:
                result = fetcher()
                values.update(result.values)
                statuses.append(
                    SourceStatus(
                        name=name,
                        ok=True,
                        detail="ok",
                        updated_at=result.updated_at,
                    )
                )
            except Exception as exc:
                statuses.append(SourceStatus(name=name, ok=False, detail=str(exc), updated_at=None))

        return values, statuses

    def _estimate_onset_window(
        self,
        event_type: str,
        risk_score: float,
        detection_confidence: float,
    ) -> OnsetWindow:
        if event_type == EventType.FLASH_FLOOD.value:
            start, end = 1.0, 6.0
        elif event_type == EventType.LONG_RAINFALL.value:
            start, end = 6.0, 24.0
        elif event_type == EventType.THAW.value:
            start, end = 12.0, 48.0
        else:
            start, end = 6.0, 36.0

        adjusted_end = end + (1.0 - min(1.0, max(risk_score, detection_confidence))) * 12.0
        return OnsetWindow(
            start_hours=start,
            end_hours=round(adjusted_end, 1),
            summary=f"Expected onset between +{start:.0f}h and +{adjusted_end:.0f}h.",
        )

    def _dominant_factor(self, detection: pd.Series | None, event_type: str) -> str:
        if detection is None:
            return event_type

        metadata_key = f"{event_type}_metadata"
        metadata = detection.get(metadata_key)
        if isinstance(metadata, dict) and metadata.get("dominant_factor"):
            return str(metadata["dominant_factor"])

        return event_type

    def _historical_comparison(
        self,
        event_type: str,
        merged: pd.DataFrame,
    ) -> HistoricalComparison:
        try:
            typed_event = EventType(event_type)
        except Exception:
            typed_event = EventType.SEASONAL_DEPENDENCY

        confidence = compute_historical_confidence(
            etype=typed_event,
            output_dir=str(self.config.paths.evaluation_json.parent),
        )

        latest_level = float(merged.iloc[-1].get("water_level_m", 0.0) or 0.0)
        water_series = merged["water_level_m"] if "water_level_m" in merged.columns else pd.Series()
        historical_levels = pd.to_numeric(water_series, errors="coerce").dropna()
        if historical_levels.empty:
            percentile = 0.5
        else:
            percentile = float((historical_levels <= latest_level).mean())

        return HistoricalComparison(
            event_type=typed_event.value,
            historical_confidence=confidence,
            percentile_rank=percentile,
            summary=(
                f"Historical confidence for {typed_event.value}: {confidence:.2f}; "
                f"current water level percentile: {percentile:.2f}."
            ),
        )

    def _persist_result(self, result: ContinuousEvaluationResult) -> None:
        latest_path = self._continuous_latest_path()
        history_path = self._continuous_history_path()

        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json())
            handle.write("\n")

    def _continuous_latest_path(self) -> Path:
        return self.config.paths.evaluation_json.parent / "continuous_latest.json"

    def _continuous_history_path(self) -> Path:
        return self.config.paths.evaluation_json.parent / "continuous_history.jsonl"

    def _resolve_stormglass_key(self) -> str | None:
        env_value = os.getenv("STORMGLASS_API_KEY")
        if env_value:
            return env_value

        env_path = Path(".env")
        if not env_path.exists():
            return None

        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("STORMGLASS_API_KEY="):
                value = line.split("=", maxsplit=1)[1].strip()
                return value or None
        return None

    def _write_forecast_report(
        self, horizon_forecasts: dict[str, HorizonEvaluation], warnings: list[str]
    ) -> ReportArtifacts | None:
        if not horizon_forecasts:
            return None

        report_dir = self.config.paths.evaluation_json.parent
        generated_at = datetime.now(UTC).strftime("%Y-%m-%d")
        markdown_path = report_dir / f"continuous_forecast_{generated_at}.md"
        plot_path = report_dir / f"continuous_forecast_{generated_at}.png"

        ensure_parent(markdown_path)
        self._write_forecast_plot(horizon_forecasts, plot_path)
        self._write_forecast_markdown(horizon_forecasts, warnings, markdown_path)

        return ReportArtifacts(
            markdown_path=str(markdown_path),
            plot_path=str(plot_path),
        )

    def _write_forecast_plot(
        self, horizon_forecasts: dict[str, HorizonEvaluation], plot_path: Path
    ) -> None:
        labels = list(horizon_forecasts.keys())
        formatted_labels = []
        for label in labels:
            h = horizon_forecasts[label]
            if h.target_timestamp:
                formatted_labels.append(f"{label}\n{h.target_timestamp.strftime('%a, %b %d')}")
            else:
                formatted_labels.append(label)

        risk_scores = [horizon_forecasts[label].risk_level_score or 0.0 for label in labels]
        confidence_scores = [horizon_forecasts[label].confidence_score or 0.0 for label in labels]
        temps = [horizon_forecasts[label].forecasted_temperature_c or 0.0 for label in labels]
        rains = [horizon_forecasts[label].forecasted_rainfall_mm or 0.0 for label in labels]
        pressures = [horizon_forecasts[label].forecasted_pressure_hpa or 0.0 for label in labels]
        winds = [horizon_forecasts[label].forecasted_wind_speed_ms or 0.0 for label in labels]

        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(16, 10), dpi=self.config.visualization.figure_dpi)
        gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.2])

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax_text = fig.add_subplot(gs[:, 1])
        ax_text.axis("off")

        # Risk & Confidence
        ax0.plot(
            formatted_labels,
            risk_scores,
            marker="o",
            markersize=8,
            linewidth=2,
            label="Risk Score",
            color="#d62728",
        )
        ax0.plot(
            formatted_labels,
            confidence_scores,
            marker="s",
            markersize=8,
            linewidth=2,
            linestyle="--",
            label="Confidence Score",
            color="#1f77b4",
        )
        ax0.set_ylim(0.0, 1.0)
        ax0.set_title("Forecast Risk and Confidence", fontsize=14, pad=10)
        ax0.set_ylabel("Score")
        ax0.legend(loc="upper left")

        # Weather (Temperature, Rainfall, Wind)
        ax1_twin = ax1.twinx()
        ax1.bar(
            formatted_labels, rains, alpha=0.6, color="#17becf", label="Rainfall (mm)", width=0.3
        )
        ax1_twin.plot(
            formatted_labels,
            temps,
            marker="^",
            markersize=8,
            linewidth=2,
            color="#ff7f0e",
            label="Temperature (°C)",
        )
        ax1_twin.plot(
            formatted_labels,
            winds,
            marker="v",
            markersize=8,
            linewidth=2,
            linestyle=":",
            color="#9467bd",
            label="Wind Speed (m/s)",
        )

        ax1.set_ylabel("Rainfall (mm)")
        ax1_twin.set_ylabel("Temp (°C) / Wind (m/s)")
        ax1.set_title("Forecasted Weather & Wind", fontsize=14, pad=10)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        # Pressure & Warnings
        ax2.plot(
            formatted_labels,
            pressures,
            marker="d",
            markersize=8,
            linewidth=2,
            color="#2ca02c",
            label="Pressure (hPa)",
        )
        ax2.set_ylabel("Pressure (hPa)")
        ax2.set_xlabel("Horizon / Date", fontsize=12)
        ax2.legend(loc="upper left")

        # Explanations Panel
        text_content = "Forecast Explanations & Risks\n"
        text_content += "=" * 40 + "\n\n"
        for label in labels:
            h = horizon_forecasts[label]
            date_str = h.target_timestamp.strftime("%Y-%m-%d") if h.target_timestamp else "Unknown"
            risk = h.predicted_risk_level or "Unknown"

            text_content += f"Horizon: {label} ({date_str})\n"
            text_content += f"Risk Level: {risk.upper()}\n"

            if h.dominant_contributing_factor and h.dominant_contributing_factor != "none":
                text_content += f"Driver: {h.dominant_contributing_factor}\n"
            if h.warnings:
                text_content += f"Warnings: {', '.join(h.warnings)}\n"

            analysis = h.event_message if h.event_message else "No critical events detected."

            import textwrap

            wrapped = textwrap.fill(analysis, width=45)
            text_content += f"Analysis:\n{wrapped}\n"
            text_content += "-" * 40 + "\n\n"

        ax_text.text(
            0.05,
            0.95,
            text_content,
            transform=ax_text.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.9, edgecolor="#dee2e6"),
        )

        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

    def _write_forecast_markdown(
        self,
        horizon_forecasts: dict[str, HorizonEvaluation],
        warnings: list[str],
        markdown_path: Path,
    ) -> None:
        lines: list[str] = []
        lines.append(f"# Continuous Forecast Report - {self.config.experiment_name}")
        lines.append("")
        lines.append(f"Generated at: {datetime.now(UTC).isoformat()}")
        lines.append("")

        if warnings:
            lines.append("## Warnings")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.append("## Horizon Summary")
        lines.append("")
        lines.append(
            "| Horizon | Risk Level | Risk Score | Confidence | Event Type | Dominant Factor | Onset Window |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for label, horizon in horizon_forecasts.items():
            onset = horizon.expected_onset_window.summary if horizon.expected_onset_window else ""
            lines.append(
                "| {label} | {risk} | {score:.2f} | {conf:.2f} | {event} | {factor} | {onset} |".format(
                    label=label,
                    risk=horizon.predicted_risk_level or "n/a",
                    score=horizon.risk_level_score or 0.0,
                    conf=horizon.confidence_score or 0.0,
                    event=horizon.event_type or "none",
                    factor=horizon.dominant_contributing_factor or "n/a",
                    onset=onset,
                )
            )

        lines.append("")
        lines.append("## Historical Comparison")
        lines.append("")
        for label, horizon in horizon_forecasts.items():
            historical = horizon.historical_comparison
            if historical is None:
                continue
            lines.append(f"### {label}")
            lines.append(historical.summary)
            lines.append("")

        lines.append("## Horizon Warnings")
        lines.append("")
        for label, horizon in horizon_forecasts.items():
            if not horizon.warnings:
                continue
            lines.append(f"### {label}")
            for warning in horizon.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        markdown_path.write_text("\n".join(lines), encoding="utf-8")


def build_service(config_path: str) -> ContinuousEvaluationService:
    """Factory helper for external callers."""

    load_project_config(config_path)
    return ContinuousEvaluationService(config_path=config_path)
