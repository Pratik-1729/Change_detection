from pathlib import Path
import sys
import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inference.predictor import Predictor
from src.inference.visualize import (plot_results,save_results,)

from src.utils.config import PROCESSED_DATASET,CHECKPOINT_DIR

def main():
    image_path = (
        PROCESSED_DATASET/"LoveDA"/"val"/"images"/"LoveDA_Val_Urban_003516_tile_0.png"
    )

    mask_path = (
        PROCESSED_DATASET/"LoveDA"/"val"/"masks"/"LoveDA_Val_Urban_003516_tile_0.png"
    )

    predictor = Predictor(
        checkpoint_path=CHECKPOINT_DIR/"best_model.pth"
    )

    image,prediction,confidence = predictor.predict(image_path)

    ground_truth = None
    if mask_path.exists():
        ground_truth = cv2.imread(str(mask_path),
                cv2.IMREAD_GRAYSCALE,
        )

    plot_results(
        image=image,
        prediction=prediction,
        confidence=confidence.squeeze(),
        ground_truth=ground_truth,

    )

    save_results(
        image=image,
        prediction=prediction,
        output_dir="outputs/predictions",
        image_name = image_path.stem,

    )

if __name__ == "__main__":
    main()
