"""Shared helpers for reading experiment report artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import read_evaluation_report, read_json


def _missing_row(path_obj: Path) -> dict[str, Any]:
    """Return a fallback row when an evaluation artifact cannot be found."""
    return {
        "experiment_name": path_obj.stem,
        "model_name": "missing",
        "task_type": "unknown",
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "brier_score": 0.0,
        "test_rows": 0,
        "best_validation_accuracy": 0.0,
        "evaluation_json": str(path_obj),
        "missing_fields": ["evaluation.json missing"],
    }


def load_evaluation_row(path: str | Path) -> dict[str, Any]:
    """Load a single evaluation artifact from a file or experiment directory."""
    path_obj = Path(path)

    if path_obj.suffix == ".json" and path_obj.name == "evaluation.json" and path_obj.exists():
        payload = read_json(path_obj)
        row = read_evaluation_report(
            path_obj
        )  # Function not available, remove or replace as needed
        row["missing_fields"] = _missing_fields(payload)
        summary_path = path_obj.with_name("training_summary.json")
        row["training_summary_json"] = str(summary_path) if summary_path.exists() else "n/a"
        return row

    if path_obj.is_dir():
        eval_path = path_obj / "evaluation.json"
        if eval_path.exists():
            payload = read_json(eval_path)
            row = read_evaluation_report(eval_path)
            row["missing_fields"] = _missing_fields(payload)
            summary_path = eval_path.with_name("training_summary.json")
            row["training_summary_json"] = str(summary_path) if summary_path.exists() else "n/a"
            return row

    return _missing_row(path_obj)


def iter_evaluation_artifacts(reports_root: str | Path) -> list[Path]:
    """Find all evaluation.json artifacts under a reports root."""
    root = Path(reports_root)
    return sorted(root.rglob("evaluation.json")) if root.exists() else []


def _missing_fields(payload: dict[str, Any]) -> list[str]:
    """Report which standard metrics are absent from the raw payload."""
    mapping = {
        "accuracy": ["accuracy", "test_accuracy"],
        "precision": ["precision", "test_precision"],
        "recall": ["recall", "test_recall"],
        "f1_score": ["f1_score", "test_f1"],
        "brier_score": ["brier_score", "test_brier"],
    }
    missing: list[str] = []
    for field, aliases in mapping.items():
        if not any(alias in payload for alias in aliases):
            missing.append(field)
    return missing
