"""
=========================================================
Synthetic Data Assurance
Upload Data
=========================================================
"""

import streamlit as st
import pandas as pd

from src.utils import (
    load_dataset,
    dataset_summary,
    column_summary,
    missing_value_summary,
)

st.set_page_config(
    page_title="Upload Data",
    page_icon="📂",
    layout="wide",
)

st.title("📂 Upload Dataset")

st.write(
    "Upload any CSV or Excel dataset to begin the synthetic data generation process."
)

# =====================================================
# Upload File
# =====================================================

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Please upload a dataset.")
    st.stop()

# =====================================================
# Load Dataset
# =====================================================

try:

    dataframe = load_dataset(uploaded_file)

except Exception as e:

    st.error(f"Unable to read dataset.\n\n{e}")

    st.stop()

# =====================================================
# Save Session
# =====================================================

st.session_state.raw_data = dataframe

# =====================================================
# Dataset Preview
# =====================================================

st.markdown("---")

st.header("Dataset Preview")

st.dataframe(
    dataframe.head(),
    use_container_width=True,
)

# =====================================================
# Dataset Summary
# =====================================================

st.markdown("---")

st.header("Dataset Summary")

summary = dataset_summary(dataframe)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Rows", summary["Rows"])

c2.metric("Columns", summary["Columns"])

c3.metric(
    "Missing Values",
    summary["Missing Values"],
)

c4.metric(
    "Duplicate Rows",
    summary["Duplicate Rows"],
)

c5.metric(
    "Memory (MB)",
     summary.get("Memory Usage (MB)", 0),
)

# =====================================================
# Column Information
# =====================================================

st.markdown("---")

st.header("Column Information")

st.dataframe(

    column_summary(dataframe),

    use_container_width=True,

)

# =====================================================
# Missing Values
# =====================================================

st.markdown("---")

st.header("Missing Value Summary")

missing = missing_value_summary(dataframe)

if missing["Missing Values"].sum() == 0:

    st.success("No missing values detected.")

else:

    st.dataframe(
        missing,
        use_container_width=True,
    )

# =====================================================
# Dataset Types
# =====================================================

st.markdown("---")

st.header("Column Types")

numeric = dataframe.select_dtypes(
    include="number"
).columns.tolist()

categorical = dataframe.select_dtypes(
    exclude="number"
).columns.tolist()

c1, c2 = st.columns(2)

with c1:

    st.subheader("Numeric")

    st.write(numeric)

with c2:

    st.subheader("Categorical")

    st.write(categorical)

# =====================================================
# Success
# =====================================================

st.markdown("---")

st.success(
    "Dataset uploaded successfully. Continue to the **Preprocessing** page."
)