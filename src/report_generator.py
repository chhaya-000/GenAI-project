"""
=========================================================
Synthetic Data Validation
PDF Report Generator
=========================================================
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.config import REPORTS_DIR


class ReportGenerator:
    """
    Generates PDF evaluation reports.
    """

    def __init__(self):

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.styles = getSampleStyleSheet()

    # =====================================================
    # Helper Table
    # =====================================================

    def create_table(self, dictionary):

        data = [["Metric", "Value"]]

        for key, value in dictionary.items():

            data.append([str(key), str(value)])

        table = Table(data, colWidths=[3.3 * inch, 2.8 * inch])

        table.setStyle(

            TableStyle(

                [

                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),

                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),

                    ("TOPPADDING", (0, 1), (-1, -1), 6),

                ]

            )

        )

        return table

    # =====================================================
    # Generate Report
    # =====================================================

    def generate(

        self,

        dataset_summary,

        training_summary,

        evaluation_results,

        filename="Synthetic_Data_Report.pdf",

    ):

        path = REPORTS_DIR / filename

        doc = SimpleDocTemplate(path)

        story = []

        title = Paragraph(

            "<b>Synthetic Data Validation Report</b>",

            self.styles["Title"],

        )

        story.append(title)

        story.append(Spacer(1, 0.25 * inch))

        story.append(

            Paragraph(

                f"Generated on: {datetime.now():%d-%m-%Y %H:%M:%S}",

                self.styles["Normal"],

            )

        )

        story.append(Spacer(1, 0.25 * inch))

        # =================================================
        # Dataset Summary
        # =================================================

        story.append(

            Paragraph(

                "<b>Dataset Summary</b>",

                self.styles["Heading2"],

            )

        )

        story.append(

            self.create_table(dataset_summary)

        )

        story.append(Spacer(1, 0.3 * inch))

        # =================================================
        # Training Summary
        # =================================================

        if training_summary:

            story.append(

                Paragraph(

                    "<b>Training Summary</b>",

                    self.styles["Heading2"],

                )

            )

            story.append(

                self.create_table(

                    training_summary

                )

            )

            story.append(

                Spacer(1, 0.3 * inch)

            )

        # =================================================
        # Statistical Fidelity
        # =================================================

        stats = evaluation_results.get(

            "statistics",

            {},

        )

        story.append(

            Paragraph(

                "<b>Statistical Fidelity</b>",

                self.styles["Heading2"],

            )

        )

        story.append(

            self.create_table(stats)

        )

        story.append(

            Spacer(1, 0.3 * inch)

        )

        # =================================================
        # ML Utility
        # =================================================

        utility = evaluation_results.get(

            "ml_utility",

            {},

        )

        story.append(

            Paragraph(

                "<b>Machine Learning Utility</b>",

                self.styles["Heading2"],

            )

        )

        story.append(

            self.create_table(

                utility

            )

        )

        story.append(

            Spacer(1, 0.3 * inch)

        )

        # =================================================
        # Privacy
        # =================================================

        privacy = evaluation_results.get(

            "privacy",

            {},

        )

        story.append(

            Paragraph(

                "<b>Privacy Metrics</b>",

                self.styles["Heading2"],

            )

        )

        story.append(

            self.create_table(

                privacy

            )

        )

        story.append(

            Spacer(1, 0.3 * inch)

        )

        # =================================================
        # SynTrust Score
        # =================================================

        story.append(

            Paragraph(

                "<b>SynTrust Score</b>",

                self.styles["Heading2"],

            )

        )

        score = evaluation_results.get(

            "syntrust_score",

            0,

        )

        story.append(

            Paragraph(

                f"<b>{score:.4f}</b>",

                self.styles["Title"],

            )

        )

        story.append(

            Spacer(1, 0.3 * inch)

        )

        # =================================================
        # AI Insights
        # =================================================

        story.append(

            Paragraph(

                "<b>AI Insights</b>",

                self.styles["Heading2"],

            )

        )

        insights = evaluation_results.get(

            "insights",

            [],

        )

        if insights:

            for insight in insights:

                story.append(

                    Paragraph(

                        f"• {insight}",

                        self.styles["Normal"],

                    )

                )

        else:

            story.append(

                Paragraph(

                    "No insights available.",

                    self.styles["Normal"],

                )

            )

        story.append(

            Spacer(1, 0.4 * inch)

        )

        # =================================================
        # Footer
        # =================================================

        story.append(

            Paragraph(

                "Generated by Synthetic Data Validation Platform",

                self.styles["Italic"],

            )

        )

        doc.build(story)

        return path
    
    