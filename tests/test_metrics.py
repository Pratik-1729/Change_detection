from pathlib import Path
import sys
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataloader import get_dataloaders
from src.models.segformer import build_segformer
from src.evaluation.metrics import evaluate_one_batch

train_loader, _ = get_dataloaders(
    processed_root=Path("datasets/processed/LoveDA"),
    batch_size=2,
)

batch = next(iter(train_loader))

model = build_segformer()

outputs = model(
    pixel_values = batch["pixel_values"],
    label = batch["labels"],
)

logits = F.interpolate(
    outputs.logits,
    size=batch["labels"].shape[-2:],
    mode="bilinear",
    align_corners=False,
)

metrics = evaluate_one_batch(
    logits,
    batch["labels"],
    num_classes=5,
)

print(metrics)