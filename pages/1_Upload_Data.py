"""
=========================================================
Synthetic Data Validation
1. Upload Dataset
=========================================================
"""

from pathlib import Path

import streamlit as st

from src.utils import (
    load_dataset,
    dataset_summary,
    preview_dataset,
)

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
    layout="wide",
)

st.title("📁 Upload Dataset")

st.markdown(
"""
Upload any **CSV** or **Excel** dataset to begin the
Synthetic Data Validation workflow.
"""
)

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:

    try:

        df = load_dataset(uploaded_file)

        # Store in session
        st.session_state.raw_data = df

        st.success("Dataset uploaded successfully.")

        st.divider()

        # =====================================================
        # Dataset Summary
        # =====================================================

        st.subheader("Dataset Summary")

        summary = dataset_summary(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", summary["Rows"])
        col2.metric("Columns", summary["Columns"])
        col3.metric("Memory (MB)", summary["Memory Usage (MB)"])

        col4, col5 = st.columns(2)

        col4.metric(
            "Missing Values",
            summary["Missing Values"],
        )

        col5.metric(
            "Duplicate Rows",
            summary["Duplicate Rows"],
        )

        st.divider()

        # =====================================================
        # Dataset Preview
        # =====================================================

        st.subheader("Preview")

        st.dataframe(
            preview_dataset(df, 10),
            use_container_width=True,
        )

        st.divider()

        # =====================================================
        # Column Information
        # =====================================================

        st.subheader("Columns")

        info = []

        for col in df.columns:

            info.append({

                "Column": col,

                "Type": str(df[col].dtype),

                "Missing": int(df[col].isna().sum()),

                "Unique": int(df[col].nunique()),

            })

        st.dataframe(
            info,
            use_container_width=True,
        )

        st.success(
            "Proceed to **2️⃣ Preprocessing** from the sidebar."
        )

    except Exception as e:

        st.error(str(e))

else:

    st.info("Please upload a CSV or Excel dataset.")

    