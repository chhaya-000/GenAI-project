"""
=========================================================
Synthetic Data Assurance
Preprocessing
=========================================================
"""

import streamlit as st

from src.preprocessing import DataPreprocessor

st.set_page_config(
    page_title="Preprocessing",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Data Preprocessing")

# =====================================================
# Check Dataset
# =====================================================

if st.session_state.raw_data is None:

    st.warning(
        "Please upload a dataset first."
    )

    st.stop()

raw_df = st.session_state.raw_data

# =====================================================
# Dataset Overview
# =====================================================

st.subheader("Original Dataset")

c1, c2, c3 = st.columns(3)

c1.metric("Rows", raw_df.shape[0])
c2.metric("Columns", raw_df.shape[1])
c3.metric(
    "Missing Values",
    int(raw_df.isna().sum().sum())
)

st.dataframe(
    raw_df.head(),
    use_container_width=True,
)

st.markdown("---")

# =====================================================
# Run Preprocessing
# =====================================================

if st.button(
    "🚀 Run Preprocessing",
    use_container_width=True,
):

    with st.spinner(
        "Preprocessing dataset..."
    ):

        preprocessor = DataPreprocessor()

        processed_df, metadata = preprocessor.preprocess(
    st.session_state.raw_data
)

        st.session_state.processed_data = processed_df
        st.session_state.metadata = metadata

        st.success(
            "Preprocessing completed successfully!"
        )

# =====================================================
# Show Results
# =====================================================

if st.session_state.processed_data is None:
    st.stop()

processed = st.session_state.processed_data

st.markdown("---")

st.header("Processed Dataset")

c1, c2, c3 = st.columns(3)

c1.metric("Rows", processed.shape[0])

c2.metric("Columns", processed.shape[1])

c3.metric(
    "Missing Values",
    int(processed.isna().sum().sum())
)

st.dataframe(
    processed.head(),
    use_container_width=True,
)

# =====================================================
# Column Types
# =====================================================

st.markdown("---")

st.header("Column Types")

numeric = processed.select_dtypes(
    include="number"
).columns.tolist()

categorical = processed.select_dtypes(
    exclude="number"
).columns.tolist()

c1, c2 = st.columns(2)

with c1:

    st.subheader("Numeric Columns")

    st.write(numeric)

with c2:

    st.subheader("Categorical Columns")

    st.write(categorical)

# =====================================================
# Metadata
# =====================================================

st.markdown("---")

st.header("Detected SDV Metadata")

metadata = st.session_state.metadata

try:

    metadata_dict = metadata.to_dict()

    st.json(metadata_dict)

except Exception:

    st.success(
        "Metadata generated successfully."
    )

# =====================================================
# Success
# =====================================================

st.markdown("---")

st.success(
    "Dataset is ready for synthetic data generation."
)