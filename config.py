"""Central configuration. Override via environment variables or a .env-style edit here."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = Path(os.environ.get("FT_DB_PATH", DATA_DIR / "finance.db"))
MODEL_PATH = Path(os.environ.get("FT_MODEL_PATH", DATA_DIR / "classifier.joblib"))

# LM Studio exposes an OpenAI-compatible server (default port 1234).
LLM_BASE_URL = os.environ.get("FT_LLM_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("FT_LLM_MODEL", "local-model")  # LM Studio uses the loaded model
LLM_TIMEOUT = float(os.environ.get("FT_LLM_TIMEOUT", "60"))

# Confidence cascade
ML_THRESHOLD = float(os.environ.get("FT_ML_THRESHOLD", "0.70"))    # below -> ask LLM
LLM_THRESHOLD = float(os.environ.get("FT_LLM_THRESHOLD", "0.60"))  # below -> manual review

# Duplicate detection
DUP_DATE_WINDOW_DAYS = int(os.environ.get("FT_DUP_WINDOW", "3"))
DUP_DESC_SIMILARITY = float(os.environ.get("FT_DUP_SIM", "0.85"))

# Minimum labeled rows needed before the ML model is trained
MIN_TRAIN_ROWS = 20
