from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT/"datasets"
LOVEDA_DIR = DATASET_DIR / "LoveDA"
LEVIR_PATH = DATASET_DIR /"LEVIR-CD"
CHECKPOINT_DIR = PROJECT_ROOT/"checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT/"logs"
REPORT_DIR = PROJECT_ROOT / "reports"
PROCESSED_DATASET = DATASET_DIR/ "processed"


NUM_CLASSES = 5
from src.utils.constants import CLASS_NAMES
IMAGE_TILE_SIZE = 512


BATCH_SIZE : int = 4
NUM_WORKERS : int = 0
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42


IMAGE_SIZE = (1024,1024)
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024


MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
PRETRAINED = True

SAVE_BEST_ONLY = True
CHECKPOINT_NAME = "segformer_best.pth"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EXPERIMENT_NAME = "Segformer_B0_LoveDA"
RUN_NAME = "Baseline"
VERSION = "v1.0"