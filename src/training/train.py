from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.segformer import build_segformer
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
from src.utils.config import (
	PROCESSED_DATASET,
	CHECKPOINT_DIR,
	EPOCHS,
	LEARNING_RATE,
	DEVICE,
	NUM_WORKERS,
	WEIGHT_DECAY,
	BATCH_SIZE
)
from src.utils.seed import set_seed
from src.utils.logging import configure_logging

import logging

logger = logging.getLogger(__name__)
def main():
    configure_logging()
    set_seed(42)
        
    # Ensure processed dataset exists
    processed_root = PROCESSED_DATASET / "LoveDA"
    if not processed_root.exists():
        logger.error("Processed dataset not found: %s", processed_root)
        raise SystemExit(1)

    # Ensure checkpoint directory exists
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create checkpoint directory: %s", CHECKPOINT_DIR)
        raise SystemExit(1)

    try:
        device = torch.device(DEVICE)
        logger.info("Using device: %s", device)

        train_loader, val_loader = get_dataloaders(
            processed_root=processed_root,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        logger.info("Train Images: %s", len(train_loader.dataset))
        logger.info("Validation Images: %s", len(val_loader.dataset))

        model = build_segformer()

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        history = trainer.fit(epochs=EPOCHS,resume=True,)
        logger.info("Training finished. History keys: %s", list(history.keys()))
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt). Exiting.")
        raise SystemExit(130)
    except Exception:
        logger.exception("Unhandled exception during training")
        raise SystemExit(1)

if __name__ == "__main__":
	main()
