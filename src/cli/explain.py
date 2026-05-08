"""CLI for SHAP explainability."""

from src.explain.feature_importance import rank_features
from src.explain.report import generate_markdown_report, save_feature_importance_csv


def explain_model(
    model,
    background_data,
    X,
    feature_names,
    output_dir,
    model_name="model",
) -> None:
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
