"""
=========================================================
Synthetic Data Validation
6. Report Generation
=========================================================
"""

from pathlib import Path

import streamlit as st

from src.report_generator import ReportGenerator
from src.utils import download_dataframe

st.set_page_config(
    page_title="Report",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Report Generation")

# =====================================================
# Check Evaluation
# =====================================================

if st.session_state.evaluation is None:

    st.warning("Please complete the evaluation first.")

    st.stop()

# =====================================================
# Dataset Information
# =====================================================

st.subheader("Project Summary")

col1, col2 = st.columns(2)

with col1:

    st.metric(

        "Rows",

        st.session_state.processed_data.shape[0],

    )

    st.metric(

        "Columns",

        st.session_state.processed_data.shape[1],

    )

with col2:

    st.metric(

        "Model",

        st.session_state.model_name,

    )

    st.metric(

        "SynTrust Score",

        round(

            st.session_state.syntrust_score,

            4,

        ),

    )

st.divider()

# =====================================================
# Generate PDF
# =====================================================

if st.button(

    "📄 Generate PDF Report",

    use_container_width=True,

):

    with st.spinner("Generating report..."):

        try:

            generator = ReportGenerator()

            dataset_summary = {

                "Rows": st.session_state.processed_data.shape[0],

                "Columns": st.session_state.processed_data.shape[1],

                "Missing Values": int(

                    st.session_state.processed_data
                    .isna()
                    .sum()
                    .sum()

                ),

                "Duplicate Rows": int(

                    st.session_state.processed_data
                    .duplicated()
                    .sum()

                ),

            }

            report_path = generator.generate(

                dataset_summary=dataset_summary,

                training_summary=st.session_state.training_summary,

                evaluation_results=st.session_state.evaluation,

            )

            st.session_state.report_path = report_path

            st.success(

                "PDF Report generated successfully."

            )

        except Exception as e:

            st.error(str(e))

# =====================================================
# Download PDF
# =====================================================

if "report_path" in st.session_state:

    report_path = Path(

        st.session_state.report_path

    )

    if report_path.exists():

        with open(

            report_path,

            "rb",

        ) as pdf:

            st.download_button(

                label="⬇ Download PDF Report",

                data=pdf,

                file_name=report_path.name,

                mime="application/pdf",

                use_container_width=True,

            )

st.divider()

# =====================================================
# Download Synthetic Dataset
# =====================================================

st.subheader("Synthetic Dataset")

download_dataframe(

    st.session_state.synthetic_data,

    "synthetic_data.csv",

)

st.divider()

# =====================================================
# Evaluation Results
# =====================================================

st.subheader("Evaluation Results")

st.json(

    st.session_state.evaluation

)

st.success(

    "Project completed successfully."

)

