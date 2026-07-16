import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.segformer import build_segformer
from src.data.dataloader import get_dataloaders

train_loader, _  = get_dataloaders(
    processed_root=Path("datasets/processed/LoveDA"),
    batch_size= 2,
)

batch = next(iter(train_loader))

model = build_segformer()
print("\n Number of classes: ")
print(model.config.num_labels) 

outputs = model(
    pixel_values = batch["pixel_values"],
    labels = batch["labels"],

)
print(batch["labels"].unique())
print("loss :",outputs.loss.item())
print("logits shape: ",outputs.logits.shape)