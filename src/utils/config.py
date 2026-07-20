from pathlib import Path
import torch
import os

IS_KAGGLE = os.path.exists("/kaggle")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT/"datasets"
LOVEDA_DIR = DATASET_DIR / "LoveDA"
LEVIR_PATH = DATASET_DIR /"LEVIR-CD"
CHECKPOINT_DIR = PROJECT_ROOT/"checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT/"logs"
REPORT_DIR = PROJECT_ROOT / "reports"

if IS_KAGGLE:
    PROCESSED_DATASET = Path(
        os.environ.get(
            "PROCESSED_DATASET",
            "/kaggle/input/datasets/pratgaiks1729/loveda-processed-tiles/processed/LoveDA",
        )
    )
else:
    PROCESSED_DATASET = Path(os.environ.get(
            "PROCESSED_DATASET",
            PROJECT_ROOT/"datasets"/"processed"/"LoveDA",
        )
    )


NUM_CLASSES = 5
from src.utils.constants import CLASS_NAMES
IMAGE_TILE_SIZE = 512


BATCH_SIZE : int = int(
    os.environ.get(
        "BATCH_SIZE",
        8 if IS_KAGGLE else 2,
    )
)
NUM_WORKERS : int = int(
    os.environ.get(
        "NUM_WORKERS",
        2 if IS_KAGGLE else 0,
    )
)
EPOCHS : int = int(
    os.environ.get(
        "EPOCHS",
        50 if IS_KAGGLE else 2,
    )
)
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42


IMAGE_SIZE = (1024,1024)
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
PRETRAINED = True

SAVE_BEST_ONLY = True
CHECKPOINT_NAME = "segformer_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EXPERIMENT_NAME = "Segformer_B0_LoveDA"
RUN_NAME = "Baseline"
VERSION = "v1.0"