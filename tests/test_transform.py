from pathlib import Path
import cv2
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.transforms import get_train_transform

image_path = Path("C:/Users/13521/Desktop/ChangeDetection/Change_detection/datasets/processed/LoveDA/Train/images/LoveDA_Train_Urban_001507_tile_2.png")
mask_path = Path("C:/Users/13521/Desktop/ChangeDetection/Change_detection/datasets/processed/LoveDA/Train/masks/LoveDA_Train_Urban_001507_tile_2.png")

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)

transform = get_train_transform()

sample = transform(
    image=image,
    mask=mask
)

image = sample["image"]
mask = sample["mask"]

print(image.shape)
print(mask.shape)

print(image.dtype)
print(mask.dtype)