"""CLI for SHAP explainability."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.cli import build_experiment, load_raw_frame
from src.events.attribution import attribute_event_type
from src.explain.feature_importance import rank_features
from src.explain.report import (
    generate_markdown_report,
    generate_visual_report,
    save_feature_importance_csv,
)
from src.utils.io import read_json


def explain_model(
    model,
    background_data,
    X,
    feature_names,
    output_dir,
    model_name="model",
) -> pd.DataFrame:
    """
    Runs the full explainability workflow:
    1. Compute SHAP values
    2. Rank features by importance
    3. Save results and generate report
    """

    from src.explain.shap_explainer import ShapAnalyzer

    analyzer = ShapAnalyzer(
        model=model,
        background_data=background_data,
    )

    shap_values = analyzer.compute_shap_values(X)

    importance_df = rank_features(
        shap_values=shap_values,
        feature_names=feature_names,
    )

    save_feature_importance_csv(
        importance_df=importance_df,
        output_path=f"{output_dir}/feature_importance.csv",
    )

    generate_markdown_report(
        importance_df=importance_df,
        output_path=f"{output_dir}/explainability_report.md",
        model_name=model_name,
    )

    # Load metrics for visual report if available
    metrics = {}
    eval_path = Path(output_dir) / "evaluation.json"
    if eval_path.exists():
        metrics = read_json(eval_path)

    generate_visual_report(
        importance_df=importance_df,
        metrics=metrics,
        output_path=f"{output_dir}/explainability_report.pdf",
        experiment_name=model_name,
        plots_dir=output_dir,
    )

    return importance_df


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the explain command."""
    parser = argparse.ArgumentParser(description="Generate SHAP explainability reports.")
    parser.add_argument("config", help="Path to project YAML config.")
    return parser


def main() -> None:
    """Parse arguments and run explainability workflow."""
    args = build_parser().parse_args()
    config, experiment = build_experiment(args.config)
    frame = load_raw_frame(config)

    # We need the processed data for SHAP
    experiment.preprocess(frame)

    # Use the preprocessed frames directly from the experiment
    train_df = experiment._train_frame
    test_df = experiment._test_frame

    if train_df is None or test_df is None:
        raise ValueError("Preprocessing did not result in valid train/test frames.")

    feature_cols = config.data.feature_columns
    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    # Use a subset for SHAP background to improve performance
    # 100-200 samples are usually sufficient for background data
    background_size = min(100, len(X_train))
    indices = np.random.choice(len(X_train), background_size, replace=False)
    X_background = X_train[indices]

    # Explain the test split
    # For reporting, we can also use a subset of test data if it's too large
    # but let's try the full test set or a representative subset
    test_size = min(500, len(X_test))
    test_indices = np.random.choice(len(X_test), test_size, replace=False)
    X_to_explain = X_test[test_indices]

    # Build model and load latest checkpoint
    model = experiment.build_model()
    experiment.load_checkpoint(model)

    output_dir = config.paths.evaluation_json.parent

    importance_df = explain_model(
        model=model,
        background_data=X_background,
        X=X_to_explain,
        feature_names=feature_cols,
        output_dir=str(output_dir),
        model_name=config.experiment_name,
    )

    # Identify and log primary event factor
    top_features = importance_df.head(5)["feature"].tolist()
    primary_factor = attribute_event_type(top_features)
    print(f"\n✓ Main factor identified: {primary_factor.value}")


if __name__ == "__main__":
    main()
