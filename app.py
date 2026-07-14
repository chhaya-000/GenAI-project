"""
=========================================================
Synthetic Data Validation Platform
Main Application
=========================================================
"""

import streamlit as st

from src.utils import initialize_session


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(

    page_title="Synthetic Data Validation",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded",

)


# =========================================================
# Initialize Session
# =========================================================

initialize_session()


# =========================================================
# Main Page
# =========================================================

st.title("📊 Synthetic Data Validation Platform")

st.markdown(
"""
Evaluate the quality of synthetic tabular datasets generated using
**CTGAN**, **TVAE**, and **Gaussian Copula**.

---

### Features

- 📁 Upload any CSV or Excel dataset
- 🧹 Automatic preprocessing
- 🤖 Synthetic data generation
- 📈 Statistical fidelity evaluation
- 🧠 ML utility assessment
- 🔒 Privacy evaluation
- ⭐ SynTrust Score
- 📄 PDF report generation

---

### Workflow

1️⃣ Upload Dataset

2️⃣ Preprocess Data

3️⃣ Generate Synthetic Data

4️⃣ Evaluate Quality

5️⃣ View Dashboard

6️⃣ Download Report

---

Use the **left sidebar** to navigate through the workflow.
"""
)

st.info(
    "This application supports any tabular dataset without requiring manual preprocessing."
)