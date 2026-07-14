"""
=========================================================
Synthetic Data Validation
Data Preprocessor
=========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sdv.metadata import SingleTableMetadata


class DataPreprocessor:

    def __init__(self):

        self.metadata = None

    # =====================================================
    # Main Preprocessing
    # =====================================================

    def preprocess(self, df: pd.DataFrame):

        data = df.copy()

        # ---------------------------------------------
        # Remove duplicate rows
        # ---------------------------------------------

        data = data.drop_duplicates()

        # ---------------------------------------------
        # Handle missing values
        # ---------------------------------------------

        for col in data.columns:

            if pd.api.types.is_numeric_dtype(data[col]):

                data[col] = data[col].fillna(
                    data[col].median()
                )

            else:

                data[col] = data[col].fillna("Missing")

        # ---------------------------------------------
        # Convert categorical columns to string
        # ---------------------------------------------

        for col in data.select_dtypes(include=["object"]).columns:

            data[col] = data[col].astype(str)

        # ---------------------------------------------
        # Convert boolean
        # ---------------------------------------------

        bool_cols = data.select_dtypes(include=["bool"]).columns

        for col in bool_cols:

            data[col] = data[col].astype(str)

        # ---------------------------------------------
        # Detect metadata
        # ---------------------------------------------

        metadata = SingleTableMetadata()

        metadata.detect_from_dataframe(data=data)

        self.metadata = metadata

        return data, metadata

    # =====================================================
    # Dataset Summary
    # =====================================================

    def summary(self, df):

        return {

            "Rows": len(df),

            "Columns": len(df.columns),

            "Missing Values": int(df.isna().sum().sum()),

            "Duplicate Rows": int(df.duplicated().sum()),

            "Memory Usage (MB)": round(

                df.memory_usage(deep=True).sum()

                / 1024

                / 1024,

                2,

            ),

        }