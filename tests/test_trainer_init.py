from pathlib import Path
import sys
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataloader import get_dataloaders
from src.models.segformer import build_segformer
from src.training.trainer import Trainer
from src.utils.config import DEVICE

device = DEVICE
train_loader,val_loader = get_dataloaders(
    processed_root=Path("datasets/processed/LoveDA")
)
model = build_segformer()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-4,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=None,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device

)

results = trainer.train_one_epoch()
print(results)