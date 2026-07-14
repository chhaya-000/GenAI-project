"""
=========================================================
Synthetic Data Validation
End-to-End Pipeline
=========================================================

Coordinates

• Data Preprocessing
• Synthetic Data Generation
• Evaluation
• Report Generation
"""

from __future__ import annotations

import logging

import pandas as pd

from src.preprocessing import DataPreprocessor
from src.synthesizers import SynthesizerManager
from src.evaluation import EvaluationEngine
from src.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class SyntheticDataPipeline:
    """
    End-to-end synthetic data workflow.
    """

    def __init__(self):

        self.preprocessor = DataPreprocessor()

        self.synthesizer = SynthesizerManager()

        self.report_generator = ReportGenerator()

        self.real_data = None

        self.processed_data = None

        self.synthetic_data = None

        self.metadata = None

        self.training_summary = None

        self.evaluation = None

    # =====================================================
    # Preprocess
    # =====================================================

    def preprocess(

        self,

        dataframe: pd.DataFrame,

    ):

        self.real_data = dataframe.copy()

        processed, metadata = self.preprocessor.preprocess(

            dataframe

        )

        self.processed_data = processed

        self.metadata = metadata

        logger.info("Preprocessing completed.")

        return processed, metadata

    # =====================================================
    # Train Synthesizer
    # =====================================================

    def train(

        self,

        model_name="CTGAN",

        epochs=300,

        batch_size=500,

    ):

        if self.processed_data is None:

            raise RuntimeError(

                "Run preprocess() first."

            )

        self.synthesizer.train_pipeline(

            dataframe=self.processed_data,

            metadata=self.metadata,

            model_name=model_name,

            epochs=epochs,

            batch_size=batch_size,

        )

        self.training_summary = (

            self.synthesizer.training_summary(

                self.processed_data

            )

        )

        logger.info("Training completed.")

        return self.training_summary

    # =====================================================
    # Generate Synthetic Dataset
    # =====================================================

    def generate(

        self,

        rows=None,

    ):

        if rows is None:

            rows = len(

                self.processed_data

            )

        self.synthetic_data = (

            self.synthesizer.sample(

                rows

            )

        )

        logger.info(

            "Synthetic dataset generated."

        )

        return self.synthetic_data

    # =====================================================
    # Evaluate
    # =====================================================

    def evaluate(

        self,

        target_column=None,

    ):

        if self.synthetic_data is None:

            raise RuntimeError(

                "Generate synthetic data first."

            )

        evaluator = EvaluationEngine(

            real_df=self.processed_data,

            synthetic_df=self.synthetic_data,

            target_column=target_column,

        )

        self.evaluation = (

            evaluator.evaluate()

        )

        logger.info(

            "Evaluation completed."

        )

        return self.evaluation

    # =====================================================
    # Report
    # =====================================================

    def create_report(

        self,

        filename="Synthetic_Data_Report.pdf",

    ):

        if self.evaluation is None:

            raise RuntimeError(

                "Run evaluation first."

            )

        dataset_summary = (

            self.preprocessor.summary(

                self.processed_data

            )

        )

        path = (

            self.report_generator.generate(

                dataset_summary=dataset_summary,

                training_summary=self.training_summary,

                evaluation_results=self.evaluation,

                filename=filename,

            )

        )

        logger.info(

            "PDF report created."

        )

        return path

    # =====================================================
    # Export Synthetic Dataset
    # =====================================================

    def export_dataset(

        self,

        filename="synthetic_data.csv",

    ):

        if self.synthetic_data is None:

            raise RuntimeError(

                "Generate synthetic data first."

            )

        return self.synthesizer.export_dataset(

            self.synthetic_data,

            filename,

        )

    # =====================================================
    # Complete Pipeline
    # =====================================================

    def run(

        self,

        dataframe,

        model_name="CTGAN",

        target_column=None,

        epochs=300,

        batch_size=500,

    ):

        self.preprocess(

            dataframe

        )

        self.train(

            model_name=model_name,

            epochs=epochs,

            batch_size=batch_size,

        )

        self.generate()

        self.evaluate(

            target_column=target_column

        )

        return self.evaluation

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        self.real_data = None

        self.processed_data = None

        self.synthetic_data = None

        self.metadata = None

        self.training_summary = None

        self.evaluation = None

        self.synthesizer.reset()

    # =====================================================
    # Properties
    # =====================================================

    @property
    def dataset_summary(self):

        if self.processed_data is None:

            return None

        return self.preprocessor.summary(

            self.processed_data

        )

    @property
    def model_summary(self):

        return self.training_summary

    @property
    def evaluation_summary(self):

        return self.evaluation

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (

            "SyntheticDataPipeline("

            f"rows={0 if self.processed_data is None else len(self.processed_data)}, "

            f"trained={self.training_summary is not None})"

        )
    
    