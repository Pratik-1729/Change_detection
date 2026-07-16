import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataloader import get_dataloaders

train_loader,val_loader = get_dataloaders(processed_root=Path("datasets/processed/LoveDA"),
                        batch_size=8,
                        )
print(f"train_batches: {len(train_loader)}")
print(f"validation_batch: {len(val_loader)}")

batch = next(iter(train_loader))

print(batch["pixel_values"].shape)
print(batch["labels"].shape)
print(batch["filename"])
print(batch["labels"].unique())