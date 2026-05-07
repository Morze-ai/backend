"""Defines the abstract experiment orchestration layer for preprocessing, training, evaluation, visualization, and inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.metrics import accuracy_score

from src.data.preprocessing import apply_preprocessor, fit_preprocessor, split_dataset
from src.training.trainer import TrainingBundle, predict_with_model, train_model
from src.utils.config import ProjectConfig
from src.utils.io import ensure_parent, read_json, write_json
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.utils.torch_runtime import prepare_torch_import
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
        split = split_dataset(
            frame=frame,
            target_column=self.config.data.target_column,
            test_size=self.config.data.test_size,
            validation_size=self.config.data.validation_size,
            random_seed=self.config.random_seed,
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
        ensure_parent(self.config.paths.training_history_csv)
        result.history.to_csv(self.config.paths.training_history_csv, index=False)
        write_json(
            self.config.paths.training_summary_json,
            {
                "experiment_name": self.config.experiment_name,
                "model_name": self.config.model.name,
                "best_validation_accuracy": result.best_validation_accuracy,
                "epochs": self.config.training.epochs,
            },
        )

    def evaluate(self) -> None:
        """Evaluates the trained model on the test set and logs performance metrics."""
        test_frame = self._test_frame if self._test_frame is not None else self._read_split("test")

        model = self._model if self._model is not None else self.build_model()
        if self._model is None:
            self.load_checkpoint(model)

        predictions, probabilities = predict_with_model(
            model=model,
            frame=test_frame,
            feature_columns=self.config.data.feature_columns,
            class_names=self.config.data.class_names,
            task_type=self._task_type(),
        )
        target_column = self.config.data.target_column
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
        ensure_parent(self.config.paths.predictions_csv)
        predictions_frame.to_csv(self.config.paths.predictions_csv, index=False)

        write_json(
            self.config.paths.evaluation_json,
            {
                "experiment_name": self.config.experiment_name,
                "model_name": self.config.model.name,
                "test_rows": len(test_frame),
                "accuracy": accuracy,
                "classes": self.config.data.class_names,
            },
        )

    def visualize_training(self) -> None:
        """Generates and saves visualizations of the training process, such as loss curves and confusion matrices."""
        history = (
            self._history
            if self._history is not None
            else pd.read_csv(self.config.paths.training_history_csv)
        )
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
        predictions, probabilities = predict_with_model(
            model=model,
            frame=transformed,
            feature_columns=self.config.data.feature_columns,
            class_names=self.config.data.class_names,
            task_type=self._task_type(),
        )
        return predictions[0], probabilities[0]

    def run(self, frame: pd.DataFrame) -> None:
        """Runs the full experiment workflow: preprocessing, training, evaluation, and visualization."""
        self.preprocess(frame)
        self.train()
        self.evaluate()
        self.visualize_training()

    def load_checkpoint(self, model: Any) -> None:
        """Loads model weights from a checkpoint file specified in the configuration."""
        prepare_torch_import()
        import torch

        state_dict = torch.load(self.config.paths.model_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

    def _processed_split_path(self, split_name: str) -> Path:
        return self.config.paths.processed_dir / f"{split_name}.csv"

    def _read_split(self, split_name: str) -> pd.DataFrame:
        path = self._processed_split_path(split_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed split '{split_name}' at {path}. Run preprocessing first."
            )
        return pd.read_csv(path)

    def _task_type(self) -> Literal["multiclass", "binary"]:
        if self.config.model.name == "logistic_regression":
            if len(self.config.data.class_names) != 2:
                raise ValueError("Binary logistic regression requires exactly two class names.")
            return "binary"
        return "multiclass"
