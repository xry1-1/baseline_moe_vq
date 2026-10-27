import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    # Environment variables are sufficient when no .env loader is installed.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(override=True)


@dataclass
class TASKS:
    PRETRAINING: str = "pre-training"
    LONG_HORIZON_FORECASTING: str = "long-horizon-forecasting"
    SHORT_HORIZON_FORECASTING: str = "short-horizon-forecasting"
    CLASSIFICATION: str = "classification"
    IMPUTATION: str = "imputation"
    ANOMALY_DETECTION: str = "anomaly-detection"
    EMBED: str = "embed"


def set_transformers_cache_path(transformers_cache_path: str):
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache_path


@dataclass
class PATHS:
    DATA_DIR: str = os.getenv("MOMENT_DATA_DIR")
    CHECKPOINTS_DIR: str = os.getenv("MOMENT_CHECKPOINTS_DIR")
    RESULTS_DIR: str = os.getenv("MOMENT_RESULTS_DIR")
    WANDB_DIR: str = os.getenv("WANDB_DIR")
