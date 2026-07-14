"""
=========================================================
Synthetic Data Validation
5. Dashboard
=========================================================
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Evaluation Dashboard")

# =====================================================
# Check Evaluation
# =====================================================

if st.session_state.evaluation is None:

    st.warning("Please complete the evaluation first.")

    st.stop()

results = st.session_state.evaluation

stats = results["statistics"]
utility = results["ml_utility"]
privacy = results["privacy"]
score = results["syntrust_score"]

# =====================================================
# KPI Cards
# =====================================================

st.subheader("Overall Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "⭐ SynTrust",
    round(score, 4),
)

c2.metric(
    "📊 Fidelity",
    stats["Statistical Fidelity"],
)

c3.metric(
    "🤖 Utility",
    utility.get("Utility Score", "N/A"),
)

c4.metric(
    "🔒 Privacy",
    privacy["Privacy Score"],
)

st.divider()

# =====================================================
# Statistical Fidelity
# =====================================================

st.subheader("📊 Statistical Fidelity")

stats_df = pd.DataFrame({

    "Metric":[

        "KS",

        "Wasserstein",

        "JS",

        "Correlation",

    ],

    "Score":[

        stats["KS Score"],

        stats["Wasserstein Score"],

        stats["JS Score"],

        stats["Correlation Score"],

    ]

})

fig, ax = plt.subplots(figsize=(7,4))

ax.bar(

    stats_df["Metric"],

    stats_df["Score"]

)

ax.set_ylim(0,1)

ax.set_ylabel("Similarity")

st.pyplot(fig)

st.divider()

# =====================================================
# Privacy
# =====================================================

st.subheader("🔒 Privacy")

privacy_df = pd.DataFrame({

    "Metric":[

        "Privacy Score",

        "NNDR",

    ],

    "Score":[

        privacy["Privacy Score"],

        privacy["NNDR"],

    ]

})

fig, ax = plt.subplots(figsize=(7,4))

ax.bar(

    privacy_df["Metric"],

    privacy_df["Score"],

)

ax.set_ylim(0,1)

st.pyplot(fig)

st.divider()

# =====================================================
# Correlation Comparison
# =====================================================

st.subheader("📈 Correlation Comparison")

real = st.session_state.processed_data.select_dtypes(
    include="number"
)

synthetic = st.session_state.synthetic_data.select_dtypes(
    include="number"
)

if real.shape[1] >= 2:

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### Real Data")

        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(
            real.corr(),
            aspect="auto"
        )

        plt.colorbar(im)

        ax.set_xticks(range(len(real.columns)))
        ax.set_xticklabels(
            real.columns,
            rotation=90,
            fontsize=8,
        )

        ax.set_yticks(range(len(real.columns)))
        ax.set_yticklabels(
            real.columns,
            fontsize=8,
        )

        st.pyplot(fig)

    with c2:

        st.markdown("### Synthetic Data")

        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(
            synthetic.corr(),
            aspect="auto"
        )

        plt.colorbar(im)

        ax.set_xticks(range(len(synthetic.columns)))
        ax.set_xticklabels(
            synthetic.columns,
            rotation=90,
            fontsize=8,
        )

        ax.set_yticks(range(len(synthetic.columns)))
        ax.set_yticklabels(
            synthetic.columns,
            fontsize=8,
        )

        st.pyplot(fig)

else:

    st.info("Not enough numeric columns.")

st.divider()

# =====================================================
# AI Insights
# =====================================================

st.subheader("💡 AI Insights")

for insight in results["insights"]:

    st.success(insight)

st.divider()

# =====================================================
# Summary Table
# =====================================================

st.subheader("Summary")

summary = pd.DataFrame({

    "Category":[

        "Statistical Fidelity",

        "ML Utility",

        "Privacy",

        "SynTrust",

    ],

    "Score":[

        stats["Statistical Fidelity"],

        utility.get(

            "Utility Score",

            None,

        ),

        privacy["Privacy Score"],

        score,

    ]

})

st.dataframe(

    summary,

    use_container_width=True,

)

st.success(

    "Proceed to **6️⃣ Report Generation**."

)