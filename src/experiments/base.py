"""Defines the abstract experiment orchestration layer for preprocessing, training, evaluation, visualization, and inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.analysis.statistical_analyzer import StatisticalAnalyzer
from src.data.feature_engineering import (
    drop_initial_lag_rows,
    engineer_features,
    generate_lag_features,
    generate_rolling_features,
    generate_seasonal_features,
)
from src.data.preprocessing import (
    apply_preprocessor,
    fit_preprocessor,
    handle_missing_values,
    split_dataset,
)
from src.events.evaluator import (
    add_temporal_columns,
    build_detector_output_frame,
    summarize_binary_event_predictions,
    summarize_by_period,
)
from src.explain.shap_explainer import ShapAnalyzer
from src.models.calibration import (
    apply_temperature_scaling,
    fit_temperature_scaling,
    load_temperature_scaling,
    save_temperature_scaling,
)
from src.training.trainer import TrainingBundle, train_model
from src.utils.config import ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.utils.torch_runtime import get_torch_device
from src.visualization.plots import save_confusion_matrix, save_training_curves


class BaseExperiment(ABC):
    """Abstract base class defining the experiment workflow for water level prediction, including data preprocessing, model training, evaluation, visualization, and inference."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = get_logger(self.config.experiment_name)
        self._train_frame: pd.DataFrame | None = None
        self._validation_frame: pd.DataFrame | None = None
        self._test_frame: pd.DataFrame | None = None
        self._history: pd.DataFrame | None = None
        self._evaluation_y_true: list[str] = []
        self._evaluation_y_pred: list[str] = []
        self._model: Any = None
        self._temperature: float = 1.0

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Returns the unique name of the experiment."""
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> Any:
        """Constructs and returns the model instance for this experiment."""
        raise NotImplementedError

    def preprocess(self, frame: pd.DataFrame) -> None:
        """Preprocesses the input DataFrame according to the experiment's configuration."""
        set_global_seed(self.config.random_seed)

        processed_frame = frame.copy()

        # Handle missing values first
        processed_frame = handle_missing_values(processed_frame)

        # Existing engineered features
        processed_frame = engineer_features(processed_frame)

        # Optional lag feature generation
        if self.config.feature_engineering.generate_lag_features:
            lag_hours = self.config.feature_engineering.lag_hours
            weather_columns = {
                "rainfall_mm": lag_hours,
                "temperature_c": lag_hours,
                "pressure_hpa": lag_hours,
            }

            available_weather = {
                col: lags for col, lags in weather_columns.items() if col in processed_frame.columns
            }

            if available_weather:
                self.logger.info(
                    f"Generating lag features for columns: {list(available_weather.keys())}"
                )

                processed_frame = generate_lag_features(
                    processed_frame,
                    timestamp_column="timestamp",
                    lag_columns=available_weather,
                    use_cuda=get_torch_device().type == "cuda",
                )

                max_lags = max(available_weather.values())

                processed_frame = drop_initial_lag_rows(
                    processed_frame,
                    max_lag_hours=max_lags,
                    timestamp_column="timestamp",
                )

                self.logger.info(
                    f"Dropped first {max_lags} rows for lag feature warmup. "
                    f"Remaining rows: {len(processed_frame)}"
                )

                # Register generated columns
                for col in available_weather:
                    for lag_hour in range(1, available_weather[col] + 1):
                        lag_col_name = f"{col}_lag_{lag_hour}h"

                        if lag_col_name not in self.config.data.feature_columns:
                            self.config.data.feature_columns.append(lag_col_name)

        # Optional rolling weather aggregations
        if self.config.feature_engineering.generate_rolling_features:
            rolling_windows = self.config.feature_engineering.rolling_windows
            rolling_base_columns = [
                column
                for column in [
                    "rainfall_mm",
                    "temperature_c",
                    "pressure_hpa",
                    "humidity_percentage",
                ]
                if column in processed_frame.columns
            ]
            if rolling_base_columns:
                processed_frame = generate_rolling_features(
                    processed_frame,
                    window_hours=rolling_windows,
                    agg_functions=["mean", "max", "min"],
                    columns_to_aggregate=rolling_base_columns,
                    timestamp_column="timestamp",
                )

                for column in rolling_base_columns:
                    for window in rolling_windows:
                        for func in ["mean", "max", "min"]:
                            feature_name = f"{column}_{func}_{window}h"
                            if (
                                feature_name in processed_frame.columns
                                and feature_name not in self.config.data.feature_columns
                            ):
                                self.config.data.feature_columns.append(feature_name)

        # Optional seasonal/calendar features
        if self.config.feature_engineering.generate_seasonal_features:
            processed_frame = generate_seasonal_features(
                processed_frame, timestamp_column="timestamp"
            )
            seasonal_numeric_features = [
                "month",
                "day_of_year",
                "day_of_week",
                "hour_of_day",
                "is_weekend",
                "is_growing_season",
                "season_code",
                "month_sin",
                "month_cos",
                "day_of_year_sin",
                "day_of_year_cos",
                "day_of_week_sin",
                "day_of_week_cos",
                "hour_of_day_sin",
                "hour_of_day_cos",
            ]
            for feature_name in seasonal_numeric_features:
                if (
                    feature_name in processed_frame.columns
                    and feature_name not in self.config.data.feature_columns
                ):
                    self.config.data.feature_columns.append(feature_name)

        split = split_dataset(
            frame=processed_frame,
            target_column=self.config.data.target_column,
            test_size=self.config.data.test_size,
            validation_size=self.config.data.validation_size,
            random_seed=self.config.random_seed,
            split_strategy=self.config.data.split_strategy,
            timestamp_column=self.config.data.timestamp_column,
            validation_start=self.config.data.validation_start,
            test_start=self.config.data.test_start,
        )

        stats = fit_preprocessor(
            frame=split.train,
            feature_columns=self.config.data.feature_columns,
            strategy=self.config.preprocessing.strategy,
        )

        self._train_frame = apply_preprocessor(split.train, self.config.data.feature_columns, stats)
        self._validation_frame = apply_preprocessor(
            split.validation, self.config.data.feature_columns, stats
        )
        self._test_frame = apply_preprocessor(split.test, self.config.data.feature_columns, stats)

        processed_dir = self.config.paths.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        self._train_frame.to_csv(self._processed_split_path("train"), index=False)
        self._validation_frame.to_csv(self._processed_split_path("validation"), index=False)
        self._test_frame.to_csv(self._processed_split_path("test"), index=False)

        write_json(self.config.paths.preprocessor_artifact, {"features": stats})

    def train(self) -> None:
        """Trains the model using the preprocessed data and training configuration."""
        set_global_seed(self.config.random_seed)
        train_frame = (
            self._train_frame if self._train_frame is not None else self._read_split("train")
        )
        validation_frame = (
            self._validation_frame
            if self._validation_frame is not None
            else self._read_split("validation")
        )

        task_type = self._task_type()
        bundle = TrainingBundle(
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_columns=self.config.data.feature_columns,
            class_names=self.config.data.class_names,
            target_column=self.config.data.target_column,
            task_type=task_type,
        )
        model = self.build_model()
        result = train_model(
            model=model,
            bundle=bundle,
            learning_rate=self.config.training.learning_rate,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            weight_decay=self.config.training.weight_decay,
            checkpoint_path=self.config.paths.model_checkpoint,
        )

        self._model = result.model
        self._history = result.history

        validation_frame = (
            self._validation_frame
            if self._validation_frame is not None
            else self._read_split("validation")
        )
        self._temperature = fit_temperature_scaling(
            model=self._model,
            frame=validation_frame,
            feature_columns=self.config.data.feature_columns,
            class_names=self.config.data.class_names,
            target_column=self.config.data.target_column,
            task_type=self._task_type(),
        )
        save_temperature_scaling(self._calibration_path(), self._temperature)

        ensure_parent(self.config.paths.training_history_csv)
        result.history.to_csv(self.config.paths.training_history_csv, index=False)
        write_json(
            self.config.paths.training_summary_json,
            {
                "experiment_name": self.config.experiment_name,
                "model_name": self.config.model.name,
                "best_validation_accuracy": result.best_validation_accuracy,
                "epochs": self.config.training.epochs,
                "temperature": self._temperature,
            },
        )

    def evaluate(self) -> None:
        """Evaluates the trained model on the test set and logs performance metrics."""
        test_frame = self._test_frame if self._test_frame is not None else self._read_split("test")

        model = self._model if self._model is not None else self.build_model()
        if self._model is None:
            self.load_checkpoint(model)

        self._temperature = load_temperature_scaling(self._calibration_path())
        self.logger.info("Applying temperature scaling and generating predictions...")
        predictions, probabilities = apply_temperature_scaling(
            model=model,
            frame=test_frame,
            feature_columns=self.config.data.feature_columns,
            class_names=self.config.data.class_names,
            task_type=self._task_type(),
            temperature=self._temperature,
        )
        target_column = self.config.data.target_column
        analysis_target_column = (
            "water_level_m" if "water_level_m" in test_frame.columns else target_column
        )
        y_true = test_frame[target_column].astype(str).tolist()
        accuracy = float(accuracy_score(y_true, predictions))
        self._evaluation_y_true = y_true
        self._evaluation_y_pred = predictions

        predictions_frame = test_frame.copy()
        predictions_frame["predicted_class"] = predictions
        predictions_frame["confidence"] = [max(values) for values in probabilities]
        for class_index, class_name in enumerate(self.config.data.class_names):
            predictions_frame[f"prob_{class_name}"] = [
                row[class_index] if class_index < len(row) else float("nan")
                for row in probabilities
            ]

        predictions_frame["model_confidence"] = predictions_frame[
            [f"prob_{class_name}" for class_name in self.config.data.class_names]
        ].max(axis=1)

        if "timestamp" in predictions_frame.columns:
            detector_frame = build_detector_output_frame(
                predictions_frame,
                timestamp_column="timestamp",
            )
            predictions_frame = pd.concat(
                [predictions_frame.reset_index(drop=True), detector_frame.reset_index(drop=True)],
                axis=1,
            )

        if "detection_confidence" in predictions_frame.columns:
            predictions_frame["ensemble_confidence"] = (
                predictions_frame["model_confidence"].fillna(0.0)
                + predictions_frame["detection_confidence"].fillna(0.0)
            ) / 2.0
        else:
            predictions_frame["ensemble_confidence"] = predictions_frame["model_confidence"]

        predictions_frame = self._attach_shap_contributing_factors(predictions_frame)

        if "timestamp" in predictions_frame.columns:
            predictions_frame = add_temporal_columns(predictions_frame, "timestamp")

        ensure_parent(self.config.paths.predictions_csv)
        predictions_frame.to_csv(self.config.paths.predictions_csv, index=False)

        evaluation_payload: dict[str, Any] = {
            "experiment_name": self.config.experiment_name,
            "model_name": self.config.model.name,
            "test_rows": len(test_frame),
            "accuracy": accuracy,
            "classes": self.config.data.class_names,
        }

        if self._task_type() == "binary" and len(self.config.data.class_names) == 2:
            positive_label = self.config.data.class_names[-1]
            probability_column = f"prob_{positive_label}"
            event_summary = summarize_binary_event_predictions(
                predictions_frame,
                target_column=target_column,
                prediction_column="predicted_class",
                positive_label=positive_label,
                timestamp_column="timestamp",
                probability_column=probability_column
                if probability_column in predictions_frame.columns
                else None,
            )
            evaluation_payload.update(event_summary)

            if (
                "timestamp" in predictions_frame.columns
                and not predictions_frame["timestamp"].isna().any()
            ):
                evaluation_payload["by_year"] = summarize_by_period(
                    predictions_frame,
                    period_column="year",
                    target_column=target_column,
                    prediction_column="predicted_class",
                    positive_label=positive_label,
                    timestamp_column="timestamp",
                    probability_column=probability_column
                    if probability_column in predictions_frame.columns
                    else None,
                )
                evaluation_payload["by_season"] = summarize_by_period(
                    predictions_frame,
                    period_column="season",
                    target_column=target_column,
                    prediction_column="predicted_class",
                    positive_label=positive_label,
                    timestamp_column="timestamp",
                    probability_column=probability_column
                    if probability_column in predictions_frame.columns
                    else None,
                )

                # Compute statistical analysis (lags, hypothesis tests, soil saturation, onset errors)
                try:
                    self.logger.info("Performing comprehensive seasonal statistical analysis...")
                    analyzer = StatisticalAnalyzer(
                        predictions_frame, dataset_name=self.config.experiment_name
                    )

                    # Detect lag columns for analysis
                    lag_columns = [
                        col
                        for col in predictions_frame.columns
                        if "_lag_" in col and col.endswith("h")
                    ]

                    # Extract onset errors from evaluation_payload if available
                    onset_errors_by_season: dict[str, list[float]] = {}
                    if "by_season" in evaluation_payload:
                        for season_data in evaluation_payload["by_season"]:
                            if "mean_onset_error_hours" in season_data:
                                season = season_data.get("period", "unknown")
                                mean_error = season_data.get("mean_onset_error_hours")
                                if mean_error and season != "unknown":
                                    onset_errors_by_season[season] = [
                                        mean_error
                                    ]  # Single value per season

                    # Generate statistical summary
                    stat_summary = analyzer.generate_statistical_summary(
                        target_column=analysis_target_column,
                        event_column="predicted_class" if positive_label else target_column,
                        soil_saturation_column="soil_saturation_index",
                        features_to_test=[
                            col
                            for col in predictions_frame.columns
                            if col.startswith(
                                ("rainfall", "temperature", "pressure", "soil_saturation")
                            )
                            and "_lag_" not in col
                        ],
                        lag_columns=lag_columns if lag_columns else None,
                        onset_errors_by_season=onset_errors_by_season
                        if onset_errors_by_season
                        else None,
                    )

                    # Convert to dict for JSON serialization
                    evaluation_payload["statistical_analysis"] = stat_summary.to_dict()
                except Exception as e:
                    # If statistical analysis fails, log warning but continue
                    print(f"Warning: Statistical analysis failed: {e}")
                    evaluation_payload["statistical_analysis"] = {"error": str(e)}

        write_json(
            self.config.paths.evaluation_json,
            evaluation_payload,
        )

    def visualize_training(self) -> None:
        """Generates and saves visualizations of the training process, such as loss curves and confusion matrices."""
        history = (
            self._history
            if self._history is not None
            else pd.read_csv(self.config.paths.training_history_csv)
        )
        self.logger.info("Generating training curves...")
        save_training_curves(
            history=history,
            output_path=self.config.paths.training_curves_png,
            dpi=self.config.visualization.figure_dpi,
        )

        if not self._evaluation_y_true or not self._evaluation_y_pred:
            predictions_frame = pd.read_csv(self.config.paths.predictions_csv)
            target_column = self.config.data.target_column
            self._evaluation_y_true = predictions_frame[target_column].astype(str).tolist()
            self._evaluation_y_pred = predictions_frame["predicted_class"].astype(str).tolist()

        self.logger.info("Generating confusion matrix...")
        save_confusion_matrix(
            y_true=self._evaluation_y_true,
            y_pred=self._evaluation_y_pred,
            class_names=self.config.data.class_names,
            output_path=self.config.paths.confusion_matrix_png,
            dpi=self.config.visualization.figure_dpi,
        )

    def predict_one(self, raw_values: dict[str, float]) -> tuple[str, list[float]]:
        """Runs inference on a single input instance and returns the predicted class and confidence scores."""
        missing_features = [
            feature for feature in self.config.data.feature_columns if feature not in raw_values
        ]
        if missing_features:
            raise ValueError(f"Missing required feature values: {missing_features}")

        preprocessor_payload = read_json(self.config.paths.preprocessor_artifact)
        raw_stats = preprocessor_payload.get("features")
        if not isinstance(raw_stats, dict):
            raise ValueError("Invalid preprocessor artifact: expected 'features' dictionary.")

        row = {feature: float(raw_values[feature]) for feature in self.config.data.feature_columns}
        frame = pd.DataFrame([row])
        transformed = apply_preprocessor(
            frame=frame,
            feature_columns=self.config.data.feature_columns,
            stats=raw_stats,
        )

        model = self.build_model()
        self.load_checkpoint(model)
        temperature = load_temperature_scaling(self._calibration_path())
        predictions, probabilities = apply_temperature_scaling(
            model=model,
            frame=transformed,
            feature_columns=self.config.data.feature_columns,
            class_names=self.config.data.class_names,
            task_type=self._task_type(),
            temperature=temperature,
        )
        return predictions[0], probabilities[0]

    def run(self, frame: pd.DataFrame) -> None:
        """Runs the full experiment workflow: preprocessing, training, evaluation, and visualization."""
        self.logger.info(f"🚀 Starting experiment: {self.config.experiment_name}")
        self.preprocess(frame)
        self.train()
        self.evaluate()
        self.visualize_training()
        self.logger.info(f"✅ Experiment {self.config.experiment_name} completed successfully")

    def load_checkpoint(self, model: Any) -> None:
        """Loads model weights from a checkpoint file specified in the configuration."""
        device = get_torch_device()
        state_dict = torch.load(self.config.paths.model_checkpoint, map_location=device)  # noqa F821 pyright: ignore[reportUndefinedVariable]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    def _processed_split_path(self, split_name: str) -> Path:
        return self.config.paths.processed_dir / f"{split_name}.csv"

    def _calibration_path(self) -> Path:
        return self.config.paths.model_checkpoint.with_name("calibration.json")

    def _attach_shap_contributing_factors(self, predictions_frame: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            return predictions_frame

        if not self.config.data.feature_columns:
            predictions_frame["contributing_factors"] = ""
            return predictions_frame

        train_frame = (
            self._train_frame if self._train_frame is not None else self._read_split("train")
        )
        feature_frame = train_frame[self.config.data.feature_columns].copy()
        if feature_frame.empty:
            predictions_frame["contributing_factors"] = ""
            return predictions_frame

        background_size = min(64, len(feature_frame))
        background = feature_frame.head(background_size).to_numpy(dtype=float, copy=True)
        shap_sample = predictions_frame[self.config.data.feature_columns].copy()
        analyzer = ShapAnalyzer(model=self._model, background_data=background)

        top_features: list[str] = []
        chunk_size = 128
        feature_names = self.config.data.feature_columns

        self.logger.info(
            f"Computing SHAP values for {len(shap_sample)} samples (chunks of {chunk_size})..."
        )
        pbar = tqdm(range(0, len(shap_sample), chunk_size), desc="SHAP Analysis", unit="chunk")
        for start_index in pbar:
            chunk = shap_sample.iloc[start_index : start_index + chunk_size]
            shap_values = analyzer.compute_shap_values(chunk.to_numpy(dtype=float, copy=True))
            if shap_values.ndim == 3:
                shap_values = np.mean(np.abs(shap_values), axis=2)
            elif shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)

            for row_values in shap_values:
                ranked = np.argsort(np.abs(row_values))[::-1]
                selected = [feature_names[index] for index in ranked[:3]]
                top_features.append(", ".join(selected))

        predictions_frame = predictions_frame.copy()
        predictions_frame["contributing_factors"] = top_features
        return predictions_frame

    def _read_split(self, split_name: str) -> pd.DataFrame:
        path = self._processed_split_path(split_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed split '{split_name}' at {path}. Run preprocessing first."
            )
        return pd.read_csv(path)

    def _task_type(self) -> Literal["multiclass", "binary"]:
        if len(self.config.data.class_names) == 2:
            return "binary"
        if self.config.model.name == "logistic_regression":
            raise ValueError("Binary logistic regression requires exactly two class names.")
        return "multiclass"
