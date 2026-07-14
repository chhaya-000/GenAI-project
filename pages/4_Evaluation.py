"""
=========================================================
Synthetic Data Validation
4. Evaluation
=========================================================
"""

import streamlit as st
import pandas as pd

from src.evaluation import EvaluationEngine

st.set_page_config(
    page_title="Evaluation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Synthetic Data Evaluation")

# =====================================================
# Check Synthetic Dataset
# =====================================================

if st.session_state.synthetic_data is None:

    st.warning("Please generate synthetic data first.")

    st.stop()

# =====================================================
# Target Selection
# =====================================================

st.subheader("Evaluation Settings")

columns = ["None"] + list(
    st.session_state.processed_data.columns
)

target = st.selectbox(

    "Target Column (Optional)",

    columns,

)

if target == "None":

    target = None

st.info(
    "If no target column is selected, only Statistical Fidelity and Privacy will be evaluated."
)

st.divider()

# =====================================================
# Run Evaluation
# =====================================================

if st.button(

    "🚀 Run Evaluation",

    use_container_width=True,

):

    with st.spinner("Evaluating synthetic dataset..."):

        try:

            evaluator = EvaluationEngine(

                real_df=st.session_state.processed_data,

                synthetic_df=st.session_state.synthetic_data,

                target_column=target,

            )

            results = evaluator.evaluate()

            st.session_state.evaluation = results

            st.session_state.syntrust_score = results[
                "syntrust_score"
            ]

            st.session_state.ai_insights = results[
                "insights"
            ]

            st.success("Evaluation completed successfully.")

        except Exception as e:

            st.error(str(e))

            st.stop()

# =====================================================
# Results
# =====================================================

if st.session_state.evaluation is not None:

    results = st.session_state.evaluation

    # ==============================================
    # SynTrust Score
    # ==============================================

    st.subheader("⭐ SynTrust Score")

    st.metric(

        "Overall Score",

        round(

            results["syntrust_score"],

            4,

        ),

    )

    st.divider()

    # ==============================================
    # Statistical Fidelity
    # ==============================================

    st.subheader("📊 Statistical Fidelity")

    stats = pd.DataFrame(

        results["statistics"].items(),

        columns=["Metric", "Value"],

    )

    st.dataframe(

        stats,

        use_container_width=True,

    )

    st.divider()

    # ==============================================
    # ML Utility
    # ==============================================

    st.subheader("🤖 Machine Learning Utility")

    utility = pd.DataFrame(

        results["ml_utility"].items(),

        columns=["Metric", "Value"],

    )

    st.dataframe(

        utility,

        use_container_width=True,

    )

    st.divider()

    # ==============================================
    # Privacy
    # ==============================================

    st.subheader("🔒 Privacy Metrics")

    privacy = pd.DataFrame(

        results["privacy"].items(),

        columns=["Metric", "Value"],

    )

    st.dataframe(

        privacy,

        use_container_width=True,

    )

    st.divider()

    # ==============================================
    # AI Insights
    # ==============================================

    st.subheader("💡 AI Insights")

    for insight in results["insights"]:

        st.success(insight)

    st.success(
        "Proceed to **5️⃣ Dashboard**."
    )