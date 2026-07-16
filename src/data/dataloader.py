from pathlib import Path
from torch.utils.data import DataLoader
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.loveda_dataset import LoveDADataset
from src.data.transforms import (get_train_transform,get_val_transform,)
from src.utils.config  import(BATCH_SIZE,NUM_CLASSES,NUM_WORKERS)


def get_dataloaders(
        processed_root: Path,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
):
    train_dataset = LoveDADataset(
        processed_root=processed_root,
        split="Train",
        transform=get_train_transform(),
    )

    val_dataset = LoveDADataset(
        processed_root=processed_root,
        split="Val",
        transform=get_val_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader,val_loader