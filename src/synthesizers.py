"""
=========================================================
Synthetic Data Validation
Synthetic Data Generation
=========================================================

Supports

• CTGAN
• TVAE
• Gaussian Copula

Compatible with any tabular dataset.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer,
)

from src.config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class SynthesizerManager:
    """
    Handles training, sampling, saving and loading
    SDV synthesizers.
    """

    MODELS = {

        "CTGAN": CTGANSynthesizer,

        "TVAE": TVAESynthesizer,

        "Gaussian Copula": GaussianCopulaSynthesizer,

    }

    def __init__(self):

        self.model = None

        self.model_name = None

        self.training_time = 0

        self.metadata = None

    # =====================================================
    # Create Model
    # =====================================================

    def create(

        self,

        model_name: str,

        metadata: SingleTableMetadata,

        epochs: int = DEFAULT_EPOCHS,

        batch_size: int = DEFAULT_BATCH_SIZE,

    ):

        if model_name not in self.MODELS:

            raise ValueError(
                f"{model_name} is not supported."
            )

        self.model_name = model_name

        self.metadata = metadata

        if model_name == "CTGAN":

            self.model = CTGANSynthesizer(

                metadata=metadata,

                epochs=epochs,

                batch_size=batch_size,

                verbose=True,

            )

        elif model_name == "TVAE":

            self.model = TVAESynthesizer(

                metadata=metadata,

                epochs=epochs,

                batch_size=batch_size,

                verbose=True,

            )

        else:

            self.model = GaussianCopulaSynthesizer(

                metadata=metadata

            )

        logger.info("%s created.", model_name)

        return self.model

    # =====================================================
    # Train
    # =====================================================

    def train(

        self,

        dataframe: pd.DataFrame,

    ):

        if self.model is None:

            raise RuntimeError(
                "Create synthesizer first."
            )

        start = time.time()

        self.model.fit(dataframe)

        self.training_time = round(

            time.time() - start,

            2,

        )

        logger.info(

            "Training finished in %.2f sec.",

            self.training_time,

        )

    # =====================================================
    # Sample
    # =====================================================

    def sample(

        self,

        num_rows: int,

    ) -> pd.DataFrame:

        if self.model is None:

            raise RuntimeError(

                "Train model first."

            )

        synthetic = self.model.sample(

            num_rows=num_rows

        )

        return synthetic

    # =====================================================
    # Sample Same Size
    # =====================================================

    def sample_same_size(

        self,

        real_df: pd.DataFrame,

    ):

        return self.sample(

            len(real_df)

        )

    # =====================================================
    # Save Model
    # =====================================================

    def save_model(

        self,

        filename=None,

    ):

        if self.model is None:

            raise RuntimeError(

                "No trained model."

            )

        MODELS_DIR.mkdir(

            exist_ok=True,

            parents=True,

        )

        if filename is None:

            filename = (

                self.model_name

                .replace(" ", "_")

                + ".pkl"

            )

        path = MODELS_DIR / filename

        joblib.dump(

            self.model,

            path,

        )

        logger.info("Model saved.")

        return path

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(

        self,

        filename,

    ):

        path = MODELS_DIR / filename

        if not path.exists():

            raise FileNotFoundError(path)

        self.model = joblib.load(path)

        logger.info("Model loaded.")

        return self.model

    # =====================================================
    # Export Dataset
    # =====================================================

    def export_dataset(

        self,

        dataframe,

        filename="synthetic_data.csv",

    ):

        OUTPUTS_DIR.mkdir(

            exist_ok=True,

            parents=True,

        )

        path = OUTPUTS_DIR / filename

        dataframe.to_csv(

            path,

            index=False,

        )

        logger.info(

            "Synthetic dataset exported."

        )

        return path

    # =====================================================
    # Training Summary
    # =====================================================

    def training_summary(

        self,

        dataframe,

    ):

        return {

            "Model": self.model_name,

            "Rows": len(dataframe),

            "Columns": dataframe.shape[1],

            "Training Time (sec)": self.training_time,

            "Synthesizer":

                type(self.model).__name__,

        }

    # =====================================================
    # Dataset Statistics
    # =====================================================

    @staticmethod
    def dataset_statistics(

        dataframe,

    ):

        return {

            "Rows":

                len(dataframe),

            "Columns":

                dataframe.shape[1],

            "Missing Values":

                int(

                    dataframe.isna()

                    .sum()

                    .sum()

                ),

            "Memory (MB)":

                round(

                    dataframe.memory_usage(

                        deep=True

                    ).sum()

                    / 1024**2,

                    2,

                ),

        }

    # =====================================================
    # Full Pipeline
    # =====================================================

    def train_pipeline(

        self,

        dataframe,

        metadata,

        model_name,

        epochs=DEFAULT_EPOCHS,

        batch_size=DEFAULT_BATCH_SIZE,

    ):

        self.create(

            model_name=model_name,

            metadata=metadata,

            epochs=epochs,

            batch_size=batch_size,

        )

        self.train(

            dataframe

        )

        return self.model

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        self.model = None

        self.model_name = None

        self.training_time = 0

        self.metadata = None

    # =====================================================
    # Available Models
    # =====================================================

    @staticmethod
    def available_models():

        MODELS_DIR.mkdir(

            exist_ok=True,

            parents=True,

        )

        return sorted(

            [

                f.name

                for f in MODELS_DIR.glob(

                    "*.pkl"

                )

            ]

        )

    # =====================================================
    # Representation
    # =====================================================

    def __repr__(self):

        return (

            f"SynthesizerManager("

            f"model={self.model_name})"

        )
    
    