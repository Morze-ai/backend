from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.cli.compare_experiments import _collect_row, command


def test_collect_row_from_json_directly(tmp_path: Path):
    """Test _collect_row when passed a direct path to evaluation.json."""
    eval_json = tmp_path / "evaluation.json"
    eval_json.write_text(
        '{"experiment_name": "json_exp", "accuracy": 0.95, "f1_score": 0.94, "task_type": "multiclass", "best_validation_accuracy": 0.96}',
        encoding="utf-8",
    )

    row = _collect_row(str(eval_json))

    assert row["experiment_name"] == "json_exp"
    assert row["accuracy"] == 0.95
    assert row["f1_score"] == 0.94
    assert row["best_validation_accuracy"] == 0.96
    assert row["task_type"] == "multiclass"
    assert row["training_summary_json"] == "n/a"


def test_collect_row_from_directory(tmp_path: Path):
    """Test _collect_row when passed a directory containing evaluation.json."""
    eval_json = tmp_path / "evaluation.json"
    eval_json.write_text(
        '{"experiment_name": "dir_exp", "model_name": "test_model", "accuracy": 0.88, "f1_score": 0.86}',
        encoding="utf-8",
    )

    row = _collect_row(str(tmp_path))

    assert row["experiment_name"] == "dir_exp"
    assert row["model_name"] == "test_model"
    assert row["accuracy"] == 0.88
    assert row["f1_score"] == 0.86


@patch("src.cli.compare_experiments.ComparisonConfig")
@patch("src.cli.compare_experiments._collect_row")
@patch("src.cli.compare_experiments.ensure_parent")
@patch("src.cli.compare_experiments.write_json")
@patch("src.cli.compare_experiments.plt")
def test_command_generates_ranking(
    mock_plt, mock_write_json, mock_ensure, mock_collect, mock_comp_config
):
    """Test that command generates ranking sorted by accuracy with all metrics."""
    # Setup
    mock_comp_config.from_yaml.return_value = MagicMock(
        experiments=[Path("e1.yaml"), Path("e2.yaml")],
        comparison_csv=Path("comp.csv"),
        comparison_json=Path("comp.json"),
        comparison_markdown=None,
        comparison_plot_png=Path("plot.png"),
        project_name="test_project",
    )

    mock_collect.side_effect = [
        {
            "experiment_name": "exp1",
            "model_name": "m1",
            "accuracy": 0.9,
            "f1_score": 0.88,
            "precision": 0.91,
            "recall": 0.87,
            "brier_score": 0.05,
            "test_rows": 200,
            "training_summary_json": "s1.json",
            "evaluation_json": "e1.json",
            "task_type": "binary",
            "best_validation_accuracy": 0.92,
        },
        {
            "experiment_name": "exp2",
            "model_name": "m2",
            "accuracy": 0.8,
            "f1_score": 0.78,
            "precision": 0.81,
            "recall": 0.77,
            "brier_score": 0.15,
            "test_rows": 200,
            "training_summary_json": "s2.json",
            "evaluation_json": "e2.json",
            "task_type": "binary",
            "best_validation_accuracy": 0.82,
        },
    ]

    # Mock plt.subplots to return a tuple (fig, ax)
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Run
    frame = command("dummy_comp.yaml")

    # Verify ranking
    assert len(frame) == 2
    assert frame.iloc[0]["experiment_name"] == "exp1"  # Ranked by accuracy (0.9 > 0.8)
    assert frame.iloc[0]["accuracy"] == 0.9
    assert frame.iloc[0]["rank"] == 1
    assert frame.iloc[1]["experiment_name"] == "exp2"
    assert frame.iloc[1]["rank"] == 2

    # Verify all metrics are present
    assert frame.iloc[0]["precision"] == 0.91
    assert frame.iloc[0]["recall"] == 0.87
    assert frame.iloc[0]["f1_score"] == 0.88
    assert frame.iloc[0]["brier_score"] == 0.05

    # Verify CSV and JSON were written
    mock_ensure.assert_called()
    mock_write_json.assert_called_once()
    assert mock_plt.subplots.called


def test_markdown_and_html_report_generation(tmp_path: Path):
    """Test that markdown and HTML reports are generated with correct structure."""
    from src.cli.compare_experiments import _generate_html_report, _generate_markdown_report
    from src.utils.config import ComparisonConfig

    # Create mock config
    config = ComparisonConfig(
        project_name="test_project",
        experiments=[],
        comparison_csv=tmp_path / "comp.csv",
        comparison_json=tmp_path / "comp.json",
        comparison_markdown=tmp_path / "comp.md",
        comparison_html=tmp_path / "comp.html",
        comparison_plot_png=tmp_path / "comp.png",
    )

    # Create test dataframe
    df = pd.DataFrame(
        {
            "experiment_name": ["exp1", "exp2"],
            "model_name": ["model_a", "model_b"],
            "accuracy": [0.95, 0.85],
            "f1_score": [0.94, 0.83],
            "precision": [0.96, 0.84],
            "recall": [0.92, 0.82],
            "brier_score": [0.05, 0.15],
            "test_rows": [200, 200],
            "training_summary_json": ["s1.json", "s2.json"],
            "evaluation_json": ["e1.json", "e2.json"],
            "missing_fields": [[], []],
            "rank": [1, 2],
        }
    )

    # Generate reports
    md_report = _generate_markdown_report(df, config)
    html_report = _generate_html_report(df, config)

    # Verify Markdown structure
    assert "# Comparison Report: test_project" in md_report
    assert "## Summary" in md_report
    assert "## Ranking" in md_report
    assert "## Detailed Metrics" in md_report
    assert "## Artifacts & Missing Data" in md_report
    assert "exp1" in md_report
    assert "exp2" in md_report
    assert "Best Experiment" in md_report

    # Verify HTML structure
    assert "<!DOCTYPE html>" in html_report
    assert "<title>Comparison Report: test_project</title>" in html_report
    assert "🏆 Best Experiment" in html_report
    assert "<h2>Ranking</h2>" in html_report
    assert "<h2>Detailed Metrics</h2>" in html_report
    assert "<h2>Artifacts & Missing Data</h2>" in html_report
    assert "exp1" in html_report
    assert "exp2" in html_report
