# backend/config.py
import os
import logging  

# --- Database Configuration (Logic for path resolution is HERE) ---
DATABASE_PATH_ENV_VAR_NAME = "VOICE_AUTH_DB_PATH"  # Name of the environment variable

# Default filename if the environment variable is not set, or if it points to a directory.
DEFAULT_DB_FILENAME = os.environ.get(
    "VOICE_AUTH_DB_FILENAME", "speaker_voice_profiles_v3.db"
)
APP_LIKE_ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Directory containing this config.py

_env_db_path_value = os.environ.get(DATABASE_PATH_ENV_VAR_NAME)
DATABASE_PATH = ""  # Initialize

if _env_db_path_value:
    _resolved_path = os.path.abspath(_env_db_path_value)
    if os.path.isdir(_resolved_path):  # If env var provided a directory
        DATABASE_PATH = os.path.join(_resolved_path, DEFAULT_DB_FILENAME)
        # print(f"CONFIG.PY: Using DB directory from env: {_resolved_path}, full path: {DATABASE_PATH}") # For debugging
    else:  # Assume env var provided a full file path
        DATABASE_PATH = _resolved_path
        # print(f"CONFIG.PY: Using DB file path from env: {DATABASE_PATH}") # For debugging
else:
    # Fallback: Use default filename in the same directory as this config.py
    DATABASE_PATH = os.path.join(APP_LIKE_ROOT_DIR, DEFAULT_DB_FILENAME)
    # print(f"CONFIG.PY: Env var {DATABASE_PATH_ENV_VAR_NAME} not set. Using default: {DATABASE_PATH}") # For debugging

# Ensure the directory for the database exists
_db_dir = os.path.dirname(DATABASE_PATH)
if _db_dir and not os.path.exists(_db_dir):
    try:
        os.makedirs(_db_dir, exist_ok=True)
        # print(f"CONFIG.PY: Created database directory: {_db_dir}") # For debugging
    except OSError as e:
        print(
            f"CRITICAL CONFIG ERROR: Could not create database directory {_db_dir}: {e}"
        )
        DATABASE_PATH = None  # Signal error to app.py

# --- Other Application Configurations (reading from environment with defaults) ---
MODEL_NAME = os.environ.get(
    "VOICE_AUTH_MODEL_NAME", "speechbrain/spkrec-ecapa-voxceleb"
)
TARGET_SR = int(os.environ.get("VOICE_AUTH_TARGET_SR", "16000"))

MIN_REQUIRED_ENROLLMENT_SAMPLES = int(
    os.environ.get("VOICE_AUTH_MIN_ENROLL_SAMPLES", "3")
)
ENROLLMENT_AUDIO_DURATION_S = int(os.environ.get("VOICE_AUTH_ENROLL_DURATION_S", "5"))

AGGREGATION_METHODS_TO_USE_AND_BENCHMARK = [
    "mean",
    "median",
    "medoid",
    "trimmed_mean",
    "max_pool",
    "min_pool",
]

DEFAULT_PRIMARY_AGGREGATION_METHOD = "mean"
PRIMARY_AGGREGATION_METHOD = os.environ.get(
    "VOICE_AUTH_PRIMARY_AGG_METHOD", DEFAULT_PRIMARY_AGGREGATION_METHOD
).lower()

ENABLE_ADAPTIVE_ENROLLMENT = (
    os.environ.get("VOICE_AUTH_ADAPTIVE_ENROLL", "True").lower() == "true"
)
ADAPTIVE_ENROLLMENT_CONFIDENCE_THRESHOLD = float(
    os.environ.get("VOICE_AUTH_ADAPTIVE_THRESHOLD", "0.70")
)
MAX_RAW_EMBEDDINGS_PER_USER = int(os.environ.get("VOICE_AUTH_MAX_RAW_EMBS", "10"))
AUTHENTICATION_THRESHOLD = float(os.environ.get("VOICE_AUTH_MAIN_THRESHOLD", "0.55"))
MIN_SNR_DB_THRESHOLD = float(os.environ.get("VOICE_AUTH_MIN_SNR_DB", "15.0"))
FLASK_PORT = int(os.environ.get("FLASK_RUN_PORT", os.environ.get("PORT", "5000")))
FLASK_DEBUG_MODE = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
FLASK_SECRET_KEY = os.environ.get(
    "FLASK_SECRET_KEY", "aryan123"
)

LOG_LEVEL = os.environ.get("VOICE_AUTH_LOG_LEVEL", "INFO").upper()

if (
    FLASK_SECRET_KEY == "aryan123"
    and not FLASK_DEBUG_MODE
):
    print(
        f"WARNING: Using default FLASK_SECRET_KEY ('{FLASK_SECRET_KEY}') in a non-debug environment. SET a proper FLASK_SECRET_KEY environment variable!"
    )