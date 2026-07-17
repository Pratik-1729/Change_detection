from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.predictor import Predictor
from src.inference.visualize import (plot_results,save_results,)

predictor = Predictor(
    "checkpoints/best_model.pth"
)

image,prediction,confidence = predictor.predict(
    Path("datasets/processed/LoveDA/val/images/LoveDA_Val_Urban_003515_tile_0.png")
)

plot_results(
    image=image,
    prediction=prediction,
    confidence=confidence.squeeze(),
)

save_results(
    image=image,
    prediction=prediction,
    output_dir="outputs/predictions",
)