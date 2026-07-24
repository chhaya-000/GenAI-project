"""
=========================================================
Synthetic Data Validation
Evaluation Engine
=========================================================

Evaluates

• Statistical Fidelity
• ML Utility
• Privacy
• SynTrust Score
• AI Insights

Compatible with any tabular dataset.
"""

from __future__ import annotations

import warnings
import logging

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class EvaluationEngine:

    """
    Complete evaluation engine.
    """

    def __init__(

        self,

        real_df,

        synthetic_df,

        target_column=None,

    ):

        self.real = real_df.copy()

        self.synthetic = synthetic_df.copy()

        self.target = target_column

        # =====================================================
    # Main Evaluation
    # =====================================================

    def evaluate(self):

        statistics = self.evaluate_statistics()

        utility = self.evaluate_ml_utility()

        privacy = self.evaluate_privacy()

        score = self.syntrust_score(

            statistics,

            utility,

            privacy,

        )

        insights = self.generate_insights(

            statistics,

            utility,

            privacy,

            score,

        )

        return {

            "statistics": statistics,

            "ml_utility": utility,

            "privacy": privacy,

            "syntrust_score": score,

            "insights": insights,

        }

        # =====================================================
    # Numeric Columns
    # =====================================================

    def numeric_columns(self):

        cols = list(

            self.real.select_dtypes(

                include=np.number

            ).columns

        )

        if self.target in cols:

            cols.remove(self.target)

        return cols

        # =====================================================
    # Categorical Columns
    # =====================================================

    def categorical_columns(self):

        cols = list(

            self.real.select_dtypes(

                exclude=np.number

            ).columns

        )

        if self.target in cols:

            cols.remove(self.target)

        return cols

        # =====================================================
    # Correlation Matrix
    # =====================================================

    def correlation_matrix(self, dataframe):

        numeric = dataframe.select_dtypes(

            include=np.number

        )

        if numeric.shape[1] < 2:

            return None

        return numeric.corr()

        # =====================================================
    # Similarity Score
    # =====================================================

    @staticmethod
    def similarity_from_distance(value):

        return max(

            0,

            1 - value,

        )

        # =====================================================
    # Statistical Fidelity
    # =====================================================

    def evaluate_statistics(self):

        numeric_cols = self.numeric_columns()

        if len(numeric_cols) == 0:

            return {

                "KS Score": 1.0,

                "Wasserstein Score": 1.0,

                "JS Score": 1.0,

                "Correlation Score": 1.0,

                "Statistical Fidelity": 1.0,

            }

        ks_scores = []

        wasserstein_scores = []

        js_scores = []

        # ---------------------------------------------
        # Column-wise Statistics
        # ---------------------------------------------

        for col in numeric_cols:

            real = self.real[col].dropna()

            synth = self.synthetic[col].dropna()

            if len(real) == 0 or len(synth) == 0:

                continue

            # =============================
            # KS Similarity
            # =============================

            ks = ks_2samp(

                real,

                synth,

            ).statistic

            ks_scores.append(

                self.similarity_from_distance(

                    ks

                )

            )

            # =============================
            # Wasserstein Similarity
            # =============================

            wd = wasserstein_distance(

                real,

                synth,

            )

            wd = wd / (

                real.std()

                + 1e-6

            )

            wasserstein_scores.append(

                self.similarity_from_distance(

                    wd

                )

            )

            # =============================
            # Jensen-Shannon Similarity
            # =============================

            hist_real, bins = np.histogram(

                real,

                bins=20,

                density=True,

            )

            hist_syn, _ = np.histogram(

                synth,

                bins=bins,

                density=True,

            )

            hist_real += 1e-8

            hist_syn += 1e-8

            js = jensenshannon(

                hist_real,

                hist_syn,

            )

            js_scores.append(

                self.similarity_from_distance(

                    js

                )

            )

        # ---------------------------------------------
        # Correlation Preservation
        # ---------------------------------------------

        corr_real = self.correlation_matrix(

            self.real

        )

        corr_syn = self.correlation_matrix(

            self.synthetic

        )

        if (

            corr_real is None

            or corr_syn is None

        ):

            corr_score = 1.0

        else:

            difference = np.abs(

                corr_real.values

                - corr_syn.values

            )

            corr_score = max(

                0,

                1 - np.nanmean(

                    difference

                ),

            )

        # ---------------------------------------------
        # Mean Scores
        # ---------------------------------------------

        ks_mean = (

            np.mean(

                ks_scores

            )

            if ks_scores

            else 1.0

        )

        wasserstein_mean = (

            np.mean(

                wasserstein_scores

            )

            if wasserstein_scores

            else 1.0

        )

        js_mean = (

            np.mean(

                js_scores

            )

            if js_scores

            else 1.0

        )

        fidelity = np.mean(

            [

                ks_mean,

                wasserstein_mean,

                js_mean,

                corr_score,

            ]

        )

        return {

            "KS Score": round(

                ks_mean,

                4,

            ),

            "Wasserstein Score": round(

                wasserstein_mean,

                4,

            ),

            "JS Score": round(

                js_mean,

                4,

            ),

            "Correlation Score": round(

                corr_score,

                4,

            ),

            "Statistical Fidelity": round(

                fidelity,

                4,

            ),

        }

        # =====================================================
    # Machine Learning Utility
    # =====================================================

    def evaluate_ml_utility(self):

        if self.target is None:

            return {

                "Task": "Not Evaluated",

                "Utility Score": None,

            }

        if self.target not in self.real.columns:

            return {

                "Task": "Target Not Found",

                "Utility Score": None,

            }

        # ---------------------------------------------
        # Prepare datasets
        # ---------------------------------------------

        real = self.real.copy()

        synthetic = self.synthetic.copy()

        X_real = real.drop(columns=[self.target])
        y_real = real[self.target]

        X_syn = synthetic.drop(columns=[self.target])
        y_syn = synthetic[self.target]

        numeric_features = list(
            X_real.select_dtypes(include=np.number).columns
        )

        categorical_features = list(
            X_real.select_dtypes(exclude=np.number).columns
        )

        # ---------------------------------------------
        # Feature Pipeline
        # ---------------------------------------------

        numeric_transformer = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

            ]

        )

        categorical_transformer = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),

            ]

        )

        preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numeric_transformer,
                    numeric_features,
                ),

                (
                    "cat",
                    categorical_transformer,
                    categorical_features,
                ),

            ]

        )

        # =================================================
        # Classification
        # =================================================

        if (

            y_real.dtype == "object"

            or str(y_real.dtype) == "category"

            or y_real.nunique() <= 20

        ):

            encoder = LabelEncoder()

            y_real = encoder.fit_transform(y_real.astype(str))

            y_syn = encoder.transform(y_syn.astype(str))

            model = Pipeline(

                [

                    (

                        "prep",

                        preprocessor,

                    ),

                    (

                        "model",

                        RandomForestClassifier(

                            n_estimators=200,

                            random_state=42,

                        ),

                    ),

                ]

            )

            model.fit(

                X_syn,

                y_syn,

            )

            predictions = model.predict(

                X_real

            )

            accuracy = accuracy_score(

                y_real,

                predictions,

            )

            precision = precision_score(

                y_real,

                predictions,

                average="weighted",

                zero_division=0,

            )

            recall = recall_score(

                y_real,

                predictions,

                average="weighted",

                zero_division=0,

            )

            f1 = f1_score(

                y_real,

                predictions,

                average="weighted",

                zero_division=0,

            )

            utility = np.mean(

                [

                    accuracy,

                    precision,

                    recall,

                    f1,

                ]

            )

            return {

                "Task": "Classification",

                "Accuracy": round(

                    accuracy,

                    4,

                ),

                "Precision": round(

                    precision,

                    4,

                ),

                "Recall": round(

                    recall,

                    4,

                ),

                "F1 Score": round(

                    f1,

                    4,

                ),

                "Utility Score": round(

                    utility,

                    4,

                ),

            }

        # =================================================
        # Regression
        # =================================================

        model = Pipeline(

            [

                (

                    "prep",

                    preprocessor,

                ),

                (

                    "model",

                    RandomForestRegressor(

                        n_estimators=200,

                        random_state=42,

                    ),

                ),

            ]

        )

        model.fit(

            X_syn,

            y_syn,

        )

        predictions = model.predict(

            X_real

        )

        r2 = r2_score(

            y_real,

            predictions,

        )

        mae = mean_absolute_error(

            y_real,

            predictions,

        )

        rmse = np.sqrt(

            mean_squared_error(

                y_real,

                predictions,

            )

        )

        utility = max(

            0,

            r2,

        )

        return {

            "Task": "Regression",

            "R²": round(

                r2,

                4,

            ),

            "MAE": round(

                mae,

                4,

            ),

            "RMSE": round(

                rmse,

                4,

            ),

            "Utility Score": round(

                utility,

                4,

            ),

        }

        # =====================================================
    # Privacy Evaluation
    # =====================================================

    def evaluate_privacy(self):

        numeric_cols = self.numeric_columns()

        if len(numeric_cols) == 0:

            return {

                "DCR": None,

                "NNDR": None,

                "Duplicate Rate": 0.0,

                "Privacy Score": 1.0,

            }

        real = self.real[numeric_cols].fillna(0)

        synthetic = self.synthetic[numeric_cols].fillna(0)

        # ---------------------------------------------
        # Distance to Closest Record (DCR)
        # ---------------------------------------------

        distances = euclidean_distances(

            synthetic,

            real,

        )

        nearest = distances.min(axis=1)

        dcr = nearest.mean()

        # ---------------------------------------------
        # Nearest Neighbour Distance Ratio (NNDR)
        # ---------------------------------------------

        sorted_dist = np.sort(

            distances,

            axis=1,

        )

        if sorted_dist.shape[1] >= 2:

            ratio = sorted_dist[:, 0] / (

                sorted_dist[:, 1] + 1e-8

            )

            nndr = ratio.mean()

        else:

            nndr = 1.0

        # ---------------------------------------------
        # Duplicate Rate
        # ---------------------------------------------

        merged = pd.merge(

            synthetic,

            self.real,

            how="inner",

        )

        duplicate_rate = (

            len(merged)

            / len(synthetic)

        )

        # ---------------------------------------------
        # Privacy Score
        # ---------------------------------------------

        privacy_score = np.mean(

            [

                1 / (1 + dcr),

                1 - duplicate_rate,

                nndr,

            ]

        )

        return {

            "DCR": round(

                dcr,

                4,

            ),

            "NNDR": round(

                nndr,

                4,

            ),

            "Duplicate Rate": round(

                duplicate_rate,

                4,

            ),

            "Privacy Score": round(

                privacy_score,

                4,

            ),

        }

    # =====================================================
    # SynTrust Score
    # =====================================================

    def syntrust_score(

        self,

        statistics,

        utility,

        privacy,

    ):

        stat = statistics.get(

            "Statistical Fidelity",

            0,

        )

        util = utility.get(

            "Utility Score",

            0,

        )

        if util is None:

            util = 0

        priv = privacy.get(

            "Privacy Score",

            0,

        )

        score = (

            0.40 * stat

            + 0.30 * util

            + 0.30 * priv

        )

        return round(

            score,

            4,

        )

    # =====================================================
    # AI Insights
    # =====================================================

    def generate_insights(

        self,

        statistics,

        utility,

        privacy,

        score,

    ):

        insights = []

        # ---------------------------------------------
        # Overall
        # ---------------------------------------------

        if score >= 0.90:

            insights.append(

                "Excellent synthetic data quality."

            )

        elif score >= 0.80:

            insights.append(

                "Very good synthetic dataset."

            )

        elif score >= 0.70:

            insights.append(

                "Acceptable quality with room for improvement."

            )

        else:

            insights.append(

                "Synthetic data quality is relatively low."

            )

        # ---------------------------------------------
        # Statistical Fidelity
        # ---------------------------------------------

        if statistics["Statistical Fidelity"] >= 0.90:

            insights.append(

                "Statistical distributions are well preserved."

            )

        else:

            insights.append(

                "Distribution mismatch detected."

            )

        # ---------------------------------------------
        # ML Utility
        # ---------------------------------------------

        util = utility.get(

            "Utility Score",

            None,

        )

        if util is not None:

            if util >= 0.90:

                insights.append(

                    "Excellent downstream ML performance."

                )

            elif util >= 0.75:

                insights.append(

                    "Synthetic data is useful for ML tasks."

                )

            else:

                insights.append(

                    "Utility for ML tasks is limited."

                )

        # ---------------------------------------------
        # Privacy
        # ---------------------------------------------

        if privacy["Privacy Score"] >= 0.90:

            insights.append(

                "Strong privacy protection."

            )

        elif privacy["Privacy Score"] >= 0.75:

            insights.append(

                "Moderate privacy protection."

            )

        else:

            insights.append(

                "Privacy risk appears elevated."

            )

        return insights

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (

            f"EvaluationEngine("

            f"rows={len(self.real)}, "

            f"columns={self.real.shape[1]})"

        )

    
        