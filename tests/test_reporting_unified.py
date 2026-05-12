import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.cli.compare_experiments import _collect_row
from src.cli.report_summary import command as report_summary_command
from src.utils.io import read_evaluation_report


def test_read_evaluation_report_standardization(tmp_path: Path):
    """Test that read_evaluation_report correctly standardizes metrics and names."""
    eval_json = tmp_path / "evaluation.json"
    payload = {
        "experiment_name": "test_exp",
        "model_name": "test_model",
        "task_type": "binary",
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.75,
        "f1_score": 0.77,
        "brier_score": 0.15,
        "test_rows": 100,
        "best_validation_accuracy": 0.88,
    }
    eval_json.write_text(json.dumps(payload), encoding="utf-8")

    report = read_evaluation_report(eval_json)

    assert report["experiment_name"] == "test_exp"
    assert report["accuracy"] == 0.85
    assert report["f1_score"] == 0.77
    assert report["evaluation_json"] == str(eval_json)
    assert "best_validation_accuracy" in report


def test_read_evaluation_report_fallback_names(tmp_path: Path):
    """Test that read_evaluation_report handles old naming conventions."""
    eval_json = tmp_path / "evaluation.json"
    payload = {
        "test_accuracy": 0.90,
        "test_f1": 0.88,
    }
    eval_json.write_text(json.dumps(payload), encoding="utf-8")

    report = read_evaluation_report(eval_json)

    assert report["accuracy"] == 0.90
    assert report["f1_score"] == 0.88


def test_report_summary_and_compare_consistency(tmp_path: Path):
    """Test that both CLI tools use the same column names and data."""
    reports_root = tmp_path / "reports"
    exp_dir = reports_root / "exp1"
    exp_dir.mkdir(parents=True)

    eval_json = exp_dir / "evaluation.json"
    payload = {
        "experiment_name": "exp1",
        "accuracy": 0.95,
        "f1_score": 0.94,
    }
    eval_json.write_text(json.dumps(payload), encoding="utf-8")

    # Test report_summary
    output_csv = tmp_path / "summary.csv"
    report_summary_command(str(reports_root), str(output_csv))

    df_summary = pd.read_csv(output_csv)
    assert "accuracy" in df_summary.columns
    assert "experiment_name" in df_summary.columns
    assert df_summary.iloc[0]["accuracy"] == 0.95

    # Test compare_experiments internal row collection
    row = _collect_row(str(eval_json))
    assert row["accuracy"] == 0.95
    assert row["experiment_name"] == "exp1"

    # Check if column names match exactly
    summary_cols = set(df_summary.columns)
    row_cols = set(row.keys())

    # common columns should be present in both
    common = {"experiment_name", "model_name", "accuracy", "f1_score", "evaluation_json"}
    for col in common:
        assert col in summary_cols
        assert col in row_cols


def test_compare_experiments_path_resolution(tmp_path: Path):
    """Test that _collect_row resolves paths (JSON, dir, YAML) correctly."""
    # 1. Directory resolution
    exp_dir = tmp_path / "exp_dir"
    exp_dir.mkdir()
    eval_json = exp_dir / "evaluation.json"
    eval_json.write_text(
        json.dumps({"experiment_name": "dir_exp", "accuracy": 0.7}), encoding="utf-8"
    )

    row = _collect_row(str(exp_dir))
    assert row["experiment_name"] == "dir_exp"
    assert row["accuracy"] == 0.7

    # 2. YAML resolution (mocking ProjectConfig)
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("dummy: config", encoding="utf-8")

    with patch("src.utils.config.ProjectConfig.from_yaml") as mock_project:
        mock_project.return_value.paths.evaluation_json = eval_json

        row = _collect_row(str(config_yaml))
        assert row["experiment_name"] == "dir_exp"
        assert row["accuracy"] == 0.7
