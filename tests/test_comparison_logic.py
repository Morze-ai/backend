from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.compare_experiments import _collect_row, command


@pytest.fixture
def mock_project_config():
    with patch("src.cli.compare_experiments.ProjectConfig") as mock:
        config = MagicMock()
        config.experiment_name = "test_exp"
        config.model.name = "test_model"
        config.paths.training_summary_json = Path("dummy_summary.json")
        config.paths.evaluation_json = Path("dummy_eval.json")
        mock.from_yaml.return_value = config
        yield mock


@pytest.fixture
def mock_read_json():
    with patch("src.cli.compare_experiments.read_json") as mock:
        yield mock


def test_collect_row_extracts_all_metrics(mock_project_config, mock_read_json):
    # Setup mocks
    mock_read_json.side_effect = [
        {"best_validation_accuracy": 0.85},  # summary
        {
            "accuracy": 0.82,
            "precision": 0.80,
            "recall": 0.78,
            "f1_score": 0.79,
            "brier_score": 0.15,
            "test_rows": 100,
            "task_type": "binary",
        },  # evaluation
    ]

    row = _collect_row("dummy_path.yaml")

    assert row["experiment"] == "test_exp"
    assert row["test_accuracy"] == 0.82
    assert row["test_precision"] == 0.80
    assert row["test_recall"] == 0.78
    assert row["test_f1"] == 0.79
    assert row["test_brier"] == 0.15
    assert row["test_rows"] == 100
    assert row["task_type"] == "binary"


def test_collect_row_from_json_directly(mock_read_json):
    # Setup mocks
    mock_read_json.return_value = {
        "experiment_name": "json_exp",
        "accuracy": 0.95,
        "f1_score": 0.94,
        "task_type": "multiclass",
        "best_validation_accuracy": 0.96,
    }

    row = _collect_row("results/evaluation.json")

    assert row["experiment"] == "json_exp"
    assert row["test_accuracy"] == 0.95
    assert row["test_f1"] == 0.94
    assert row["best_validation_accuracy"] == 0.96
    assert row["task_type"] == "multiclass"
    assert row["training_summary_json"] == "n/a"


@patch("src.cli.compare_experiments.ComparisonConfig")
@patch("src.cli.compare_experiments._collect_row")
@patch("src.cli.compare_experiments.ensure_parent")
@patch("src.cli.compare_experiments.write_json")
@patch("src.cli.compare_experiments.plt")
def test_command_generates_ranking(
    mock_plt, mock_write_json, mock_ensure, mock_collect, mock_comp_config
):
    # Setup
    mock_comp_config.from_yaml.return_value = MagicMock(
        experiments=[Path("e1.yaml"), Path("e2.yaml")],
        comparison_csv=Path("comp.csv"),
        comparison_json=Path("comp.json"),
        comparison_plot_png=Path("plot.png"),
    )

    mock_collect.side_effect = [
        {
            "experiment": "exp1",
            "model": "m1",
            "test_accuracy": 0.9,
            "test_f1": 0.88,
            "test_precision": 0.91,
            "test_recall": 0.87,
            "test_brier": 0.05,
            "test_rows": 200,
            "training_summary_json": "s1.json",
            "evaluation_json": "e1.json",
        },
        {
            "experiment": "exp2",
            "model": "m2",
            "test_accuracy": 0.8,
            "test_f1": 0.78,
            "test_precision": 0.81,
            "test_recall": 0.77,
            "test_brier": 0.15,
            "test_rows": 200,
            "training_summary_json": "s2.json",
            "evaluation_json": "e2.json",
        },
    ]

    # Mock plt.subplots to return a tuple (fig, ax)
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Run
    frame = command("dummy_comp.yaml")

    # Verify
    assert len(frame) == 2
    assert frame.iloc[0]["experiment"] == "exp1"  # Ranked by accuracy
    assert frame.iloc[0]["rank"] == 1
    assert frame.iloc[1]["rank"] == 2

    mock_write_json.assert_called_once()
    assert mock_plt.subplots.called
