import os
from pathlib import Path
from typing import Optional


def load_env_from_dotenv(dotenv_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file without overriding existing
    process environment values.
    """
    if dotenv_path:
        env_file = Path(dotenv_path).expanduser()
    else:
        env_file = Path(__file__).resolve().parent.parent / ".env"

    if not env_file.exists() or not env_file.is_file():
        return

    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        os.environ.setdefault(key, value)


load_env_from_dotenv(os.getenv("DTAO_ENV_FILE"))

# Bittensor related configuration
SUBTENSOR_NETWORK = os.getenv("SUBTENSOR_NETWORK", "finney")  # Options: finney, local, mock

# Taostats API related configuration
TAOSTATS_API_URL = os.getenv("TAOSTATS_API_URL", "https://api.taostats.io")
TAOSTATS_API_KEY = os.getenv("TAOSTATS_API_KEY", "your_api_key")

# dTAO price prediction related configuration
PREDICTION_WINDOW = 30  # Predict for future 30 days
HISTORICAL_WINDOW = 60  # Use past 60 days of data for training
MAX_SUBNETS = 150  # Adjusted to handle 100+ subnets with room for expansion

# Interface configuration
DEFAULT_DISPLAY_LIMIT = 20  # Default to display top 20 subnets
SEARCH_ENABLED = True  # Enable search functionality
ENABLE_CACHING = True  # Enable data caching
CACHE_TIMEOUT = 300  # Cache timeout in seconds

# Model configuration
MODEL_CONFIGS = {
    'lstm': {
        'units': 50,
        'dropout': 0.2, 
        'epochs': 50,
        'batch_size': 32,
        'time_steps': 7
    },
    'arima': {
        'order': (5, 1, 0)
    },
    'xgboost': {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'num_boost_round': 100
    },
    'prophet': {
        'daily_seasonality': True
    }
}

# Default prediction model
DEFAULT_MODEL = 'random_forest' 
