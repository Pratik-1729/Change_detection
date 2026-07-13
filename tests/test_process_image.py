import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.process_image import process_image

result = process_image(
    image_path=Path("datasets/LoveDA/Train/Urban/images_png/1366.png"),
    mask_path=Path("datasets/LoveDA/Train/Urban/masks_png/1366.png"),
    image_output_dir=Path("datasets/processed/LoveDA/Train/images"),
    mask_output_dir= Path("datasets/processed/LoveDA/Train/masks"),
    dataset_name="LoveDA",
    split="Train",
    region="Urban"
)
print(result)