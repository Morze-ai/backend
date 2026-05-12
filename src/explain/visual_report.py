"""Premium PDF report generation for experiment results and SHAP explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.events.attribution import attribute_event_type
from src.visualization.plots import save_confusion_matrix


class VisualReportGenerator:
    """Generates premium visual PDF reports summarizing findings and SHAP values."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        self.styles.add(
            ParagraphStyle(
                name="PremiumTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1A237E"),
                spaceAfter=30,
                alignment=1,  # Center
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="PremiumHeading",
                parent=self.styles["Heading2"],
                fontSize=18,
                textColor=colors.HexColor("#283593"),
                spaceBefore=20,
                spaceAfter=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetricLabel",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.grey,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetricValue",
                parent=self.styles["Normal"],
                fontSize=14,
                textColor=colors.black,
                fontName="Helvetica-Bold",
            )
        )

    def generate_report(
        self,
        experiment_name: str,
        metrics: dict[str, Any],
        importance_df: pd.DataFrame,
        plots_dir: Path,
    ) -> None:
        """Assembles the PDF report."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(self.output_path), pagesize=A4)
        story: list[Any] = []

        # 1. Title
        story.append(
            Paragraph(f"Experiment Analysis: {experiment_name}", self.styles["PremiumTitle"])
        )
        story.append(Spacer(1, 0.2 * inch))

        # 2. Executive Summary Metrics
        story.append(Paragraph("Executive Summary", self.styles["PremiumHeading"]))

        # Create a metrics grid
        metric_data = [
            [
                Paragraph("Accuracy", self.styles["MetricLabel"]),
                Paragraph("Precision", self.styles["MetricLabel"]),
                Paragraph("Recall", self.styles["MetricLabel"]),
                Paragraph("F1-Score", self.styles["MetricLabel"]),
            ],
            [
                Paragraph(f"{metrics.get('accuracy', 0.0):.2%}", self.styles["MetricValue"]),
                Paragraph(f"{metrics.get('precision', 0.0):.2%}", self.styles["MetricValue"]),
                Paragraph(f"{metrics.get('recall', 0.0):.2%}", self.styles["MetricValue"]),
                Paragraph(f"{metrics.get('f1_score', 0.0):.2%}", self.styles["MetricValue"]),
            ],
        ]

        if "mean_event_confidence" in metrics:
            metric_data[0].append(Paragraph("Avg Event Confidence", self.styles["MetricLabel"]))
            metric_data[1].append(
                Paragraph(
                    f"{metrics.get('mean_event_confidence', 0.0):.1%}", self.styles["MetricValue"]
                )
            )

        metric_table = Table(metric_data, colWidths=[1.2 * inch] * len(metric_data[0]))
        metric_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
                    ("TOPPADDING", (0, 1), (-1, 1), 0),
                ]
            )
        )
        story.append(metric_table)
        story.append(Spacer(1, 0.4 * inch))

        # 3. SHAP Findings
        story.append(Paragraph("SHAP Feature Importance", self.styles["PremiumHeading"]))
        story.append(
            Paragraph(
                "The following chart shows the most influential factors contributing to high water level predictions. "
                "Higher importance values indicate a stronger impact on the model's decision.",
                self.styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Generate a nice bar chart for the report
        importance_plot_path = plots_dir / "report_importance.png"
        self._create_importance_plot(importance_df.head(15), importance_plot_path)

        img = Image(str(importance_plot_path), width=6 * inch, height=4 * inch)
        story.append(img)
        story.append(Spacer(1, 0.4 * inch))

        # 4. Seasonal Performance (if available)
        if "by_season" in metrics:
            story.append(Paragraph("Seasonal Performance", self.styles["PremiumHeading"]))
            seasonal_data = [["Season", "Accuracy", "Recall", "Avg Confidence"]]
            for season_info in metrics["by_season"]:
                seasonal_data.append(
                    [
                        season_info.get("season", "unknown"),
                        f"{season_info.get('row_accuracy', 0.0):.2%}",
                        f"{season_info.get('event_recall', 0.0):.2%}",
                        f"{season_info.get('mean_event_confidence', 0.0):.1%}"
                        if season_info.get("mean_event_confidence")
                        else "N/A",
                    ]
                )

            seasonal_table = Table(seasonal_data, colWidths=[1.5 * inch] * 4)
            seasonal_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAF6")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            story.append(seasonal_table)

        # 5. Factor Attribution (O1-O4)
        story.append(Spacer(1, 0.4 * inch))
        story.append(Paragraph("Factor Attribution & Analysis", self.styles["PremiumHeading"]))

        top_features = importance_df.head(5)["feature"].tolist()
        attributed_type = attribute_event_type(top_features)
        from src.events.attribution import compute_historical_confidence

        hist_conf = compute_historical_confidence(attributed_type, output_dir=str(plots_dir))

        story.append(
            Paragraph(
                f"Based on the most influential predictive factors, the current model profile is primarily driven by: <b>{attributed_type.value}</b> (Historical Confidence: {hist_conf:.1%}).",
                self.styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # 6. Detailed Visualizations (Page 2)
        story.append(PageBreak())
        story.append(Paragraph("Detailed Performance Analysis", self.styles["PremiumTitle"]))

        # Confusion Matrix
        story.append(Paragraph("Confusion Matrix", self.styles["PremiumHeading"]))
        if "confusion_matrix" in metrics:
            cm_path = plots_dir / "report_cm.png"
            cm_data = metrics["confusion_matrix"]
            save_confusion_matrix(
                y_true=cm_data["y_true"],
                y_pred=cm_data["y_pred"],
                class_names=cm_data.get("class_names", ["Low", "High"]),
                output_path=cm_path,
                dpi=150,
            )
            story.append(Image(str(cm_path), width=4 * inch, height=4 * inch))

        # Feature Distributions (if raw data path is in metrics/config)
        # For now, we assume histograms were pre-generated if needed
        hist_path = plots_dir / "feature_histograms.png"
        if hist_path.exists():
            story.append(PageBreak())
            story.append(Paragraph("Feature Distributions", self.styles["PremiumHeading"]))
            story.append(Image(str(hist_path), width=7 * inch, height=8 * inch))

        doc.build(story)

    def _create_importance_plot(self, df: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        # Take top 15 features
        plot_df = df.sort_values("importance", ascending=False).head(15)

        sns.barplot(
            x="importance",
            y="feature",
            data=plot_df,
            palette="viridis",
            hue="feature",
            legend=False,
        )

        plt.title("Top 15 Predictive Factors (SHAP)", fontsize=16, color="#1A237E")
        plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
