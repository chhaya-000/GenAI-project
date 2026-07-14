"""
=========================================================
Synthetic Data Validation
Configuration
=========================================================
"""

from pathlib import Path
import logging

# =========================================================
# Project Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# Create folders automatically
for folder in [
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    ASSETS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# Supported Files
# =========================================================

SUPPORTED_FILE_TYPES = [
    "csv",
    "xlsx",
    "xls",
]

# =========================================================
# Default Training Parameters
# =========================================================

DEFAULT_MODEL = "CTGAN"

DEFAULT_EPOCHS = 300

DEFAULT_BATCH_SIZE = 500

DEFAULT_SAMPLE_SIZE = 1000

RANDOM_STATE = 42

# =========================================================
# Available Synthesizers
# =========================================================

SUPPORTED_MODELS = [

    "CTGAN",

    "TVAE",

    "Gaussian Copula",

]

# =========================================================
# Evaluation Weights
# =========================================================

SYNTRUST_WEIGHTS = {

    "statistical_fidelity": 0.40,

    "ml_utility": 0.30,

    "privacy": 0.30,

}

# =========================================================
# Privacy Thresholds
# =========================================================

LOW_PRIVACY_THRESHOLD = 0.30

MEDIUM_PRIVACY_THRESHOLD = 0.60

HIGH_PRIVACY_THRESHOLD = 0.80

# =========================================================
# Visualization
# =========================================================

MAX_PLOT_COLUMNS = 8

MAX_CORRELATION_COLUMNS = 20

# =========================================================
# Logging
# =========================================================

LOG_LEVEL = logging.INFO

logging.basicConfig(

    level=LOG_LEVEL,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger("SyntheticDataValidation")