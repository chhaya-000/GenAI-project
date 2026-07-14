"""
=========================================================
Synthetic Data Validation
Utility Functions
=========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.config import (
    OUTPUTS_DIR,
    SUPPORTED_FILE_TYPES,
)

# =========================================================
# Session State
# =========================================================

def initialize_session():

    defaults = {

        "raw_data": None,

        "processed_data": None,

        "metadata": None,

        "synthetic_data": None,

        "evaluation": None,

        "syntrust_score": None,

        "ai_insights": None,

        "model_name": None,

        "training_summary": None,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# =========================================================
# Dataset Loading
# =========================================================

def load_dataset(uploaded_file) -> pd.DataFrame:

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":

        return pd.read_csv(uploaded_file)

    elif suffix in [".xlsx", ".xls"]:

        return pd.read_excel(uploaded_file)

    raise ValueError(

        f"Supported file types are {SUPPORTED_FILE_TYPES}"

    )


# =========================================================
# Save Dataset
# =========================================================

def save_dataset(

    dataframe: pd.DataFrame,

    filename: str,

):

    OUTPUTS_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )

    path = OUTPUTS_DIR / filename

    dataframe.to_csv(

        path,

        index=False,

    )

    return path


# =========================================================
# Dataset Summary
# =========================================================

def dataset_summary(df: pd.DataFrame) -> Dict:

    return {

        "Rows": len(df),

        "Columns": df.shape[1],

        "Missing Values": int(

            df.isna().sum().sum()

        ),

        "Duplicate Rows": int(

            df.duplicated().sum()

        ),

        "Memory Usage (MB)": round(

            df.memory_usage(deep=True).sum()

            / (1024 ** 2),

            2,

        ),

    }


# =========================================================
# Missing Values
# =========================================================

def missing_value_summary(

    df: pd.DataFrame,

):

    summary = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isna().sum().values,

        "Missing (%)":

            (df.isna().mean() * 100).round(2).values,

    })

    return summary.sort_values(

        "Missing Values",

        ascending=False,

    )


# =========================================================
# Duplicate Summary
# =========================================================

def duplicate_summary(df):

    duplicates = int(

        df.duplicated().sum()

    )

    return {

        "Duplicate Rows": duplicates,

        "Duplicate Percentage":

            round(

                duplicates / len(df) * 100,

                2,

            ),

    }


# =========================================================
# Column Information
# =========================================================

def column_summary(

    df: pd.DataFrame,

):

    return pd.DataFrame({

        "Column": df.columns,

        "Data Type":

            df.dtypes.astype(str),

        "Unique Values":

            df.nunique(),

        "Missing Values":

            df.isna().sum(),

    }).reset_index(

        drop=True

    )


# =========================================================
# Numeric Columns
# =========================================================

def numeric_columns(

    df: pd.DataFrame,

) -> List[str]:

    return list(

        df.select_dtypes(

            include="number"

        ).columns

    )


# =========================================================
# Categorical Columns
# =========================================================

def categorical_columns(

    df: pd.DataFrame,

) -> List[str]:

    return list(

        df.select_dtypes(

            exclude="number"

        ).columns

    )


# =========================================================
# Dataset Preview
# =========================================================

def preview_dataset(

    df: pd.DataFrame,

    rows: int = 5,

):

    return df.head(rows)


# =========================================================
# Download DataFrame
# =========================================================

def download_dataframe(

    dataframe: pd.DataFrame,

    filename: str,

):

    st.download_button(

        label="⬇ Download CSV",

        data=dataframe.to_csv(

            index=False

        ).encode("utf-8"),

        file_name=filename,

        mime="text/csv",

        use_container_width=True,

    )


# =========================================================
# Streamlit Messages
# =========================================================

def success(message):

    st.success(message)


def warning(message):

    st.warning(message)


def error(message):

    st.error(message)


def info(message):

    st.info(message)