from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.predictor import Predictor

predictor  = Predictor(
    checkpoint_path=Path("checkpoints/best_model.pth")
)

image,prediction,confidence = predictor.predict(Path("datasets/processed/LoveDA/val/images/LoveDA_Val_Urban_003515_tile_0.png")
)

print(image.shape)
print(prediction.shape)
print(confidence.shape)

print("Labels: ",set(prediction.flatten()))