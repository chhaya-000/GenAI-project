"""
=========================================================
Synthetic Data Validation
3. Synthetic Data Generation
=========================================================
"""

import streamlit as st

from src.synthesizers import SynthesizerManager
from src.config import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    SUPPORTED_MODELS,
)

st.set_page_config(
    page_title="Synthetic Data Generation",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Synthetic Data Generation")

# =====================================================
# Check preprocessing
# =====================================================

if st.session_state.processed_data is None:

    st.warning("Please preprocess the dataset first.")

    st.stop()

# =====================================================
# Dataset Information
# =====================================================

st.subheader("Processed Dataset")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    st.session_state.processed_data.shape[0],
)

col2.metric(
    "Columns",
    st.session_state.processed_data.shape[1],
)

col3.metric(
    "Missing Values",
    int(
        st.session_state.processed_data
        .isna()
        .sum()
        .sum()
    ),
)

st.divider()

# =====================================================
# Model Selection
# =====================================================

st.subheader("Training Configuration")

model_name = st.selectbox(

    "Synthesizer",

    SUPPORTED_MODELS,

)

epochs = DEFAULT_EPOCHS

batch_size = DEFAULT_BATCH_SIZE

if model_name != "Gaussian Copula":

    col1, col2 = st.columns(2)

    with col1:

        epochs = st.number_input(

            "Epochs",

            min_value=10,

            max_value=5000,

            value=DEFAULT_EPOCHS,

            step=10,

        )

    with col2:

        batch_size = st.number_input(

            "Batch Size",

            min_value=50,

            max_value=5000,

            value=DEFAULT_BATCH_SIZE,

            step=50,

        )

rows = st.number_input(

    "Synthetic Rows",

    min_value=1,

    value=len(st.session_state.processed_data),

)

st.divider()

# =====================================================
# Generate Synthetic Data
# =====================================================

if st.button(

    "🚀 Train & Generate",

    use_container_width=True,

):

    with st.spinner("Training synthesizer..."):

        try:

            manager = SynthesizerManager()

            manager.create(

                model_name=model_name,

                metadata=st.session_state.metadata,

                epochs=epochs,

                batch_size=batch_size,

            )

            manager.create(

                model_name=model_name,

                metadata=st.session_state.metadata,

                epochs=epochs,

                batch_size=batch_size,

            )

# ======================================
# DEBUG
# ======================================

            st.write("### Debug Information")

            st.write("Shape:")
            st.write(st.session_state.processed_data.shape)

            st.write("Columns:")
            st.write(st.session_state.processed_data.columns.tolist())

            st.write("Data Types:")
            st.write(st.session_state.processed_data.dtypes)

            st.write("Unique Values:")
            st.write(st.session_state.processed_data.nunique())

            st.write("First 5 Rows:")
            st.dataframe(st.session_state.processed_data.head())

            manager.train(

                st.session_state.processed_data

            )

            

            synthetic = manager.sample(rows)

            st.session_state.synthetic_data = synthetic

            st.session_state.training_summary = (

                manager.training_summary(

                    st.session_state.processed_data

                )

            )

            st.session_state.model_name = model_name

            manager.save_model()

            manager.export_dataset(

                synthetic,

                "synthetic_data.csv",

            )

            st.success(

                "Synthetic dataset generated successfully."

            )

        except Exception as e:

            st.error(str(e))

            st.stop()

# =====================================================
# Results
# =====================================================

if st.session_state.synthetic_data is not None:

    st.subheader("Training Summary")

    summary = st.session_state.training_summary

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Model",

            summary["Model"],

        )

        st.metric(

            "Rows",

            summary["Rows"],

        )

    with col2:

        st.metric(

            "Columns",

            summary["Columns"],

        )

        st.metric(

            "Training Time (sec)",

            summary["Training Time (sec)"],

        )

    st.divider()

    st.subheader("Synthetic Dataset Preview")

    st.dataframe(

        st.session_state.synthetic_data.head(),

        use_container_width=True,

    )

    st.download_button(

        label="⬇ Download Synthetic Dataset",

        data=st.session_state.synthetic_data.to_csv(

            index=False

        ),

        file_name="synthetic_data.csv",

        mime="text/csv",

        use_container_width=True,

    )

    st.success(

        "Proceed to **4️⃣ Evaluation**."

    )

