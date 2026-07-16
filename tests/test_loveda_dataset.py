from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loveda_dataset import LoveDADataset
from src.data.transforms import get_train_transform

dataset = LoveDADataset(
    processed_root= Path("datasets/processed/LoveDA"),
    split="Train",
    transform=get_train_transform(),
    )
print("Dset size: ", len(dataset))
sample = dataset[0]

print("Image_shape: ",sample["pixel_values"].shape)
print("Mask_size: ",sample["labels"].shape)
print(sample["filename"])