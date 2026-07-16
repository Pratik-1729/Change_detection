from pathlib import Path
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.evaluation.metrics import SegmentationMetrics
from src.utils.config import(
    CHECKPOINT_DIR,
    NUM_CLASSES,
)

class Trainer:
    def __init__(
            self,
            model,
            optimizer: Optimizer,
            scheduler: LRScheduler | None,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
    ):
         self.model = model
         self.optimizer = optimizer
         self.scheduler = scheduler
         self.train_loader = train_loader
         self.val_loader = val_loader
         self.device = device

         self.model.to(self.device)
         self.num_classes = NUM_CLASSES

         self.train_metrics = SegmentationMetrics(self.num_classes)
         self.val_metrics = SegmentationMetrics(self.num_classes)
         
         self.history = {
             "train_loss":[],
             "val_loss":[],
             "train_miou":[],
             "val_miou":[],
             "train_dice":[],
             "val_dice":[],
             "train_pixel_accuracy":[],
             "val_pixel_accuracy":[],
        }
         
         self.best_miou = 0.0

         self.patience = 10
         self.counter = 0


         self.checkpoint_dir = CHECKPOINT_DIR
         self.checkpoint_dir.mkdir(
              parents=True,
              exist_ok=True,
         )
         print("="*60)
         print("Trainer Initialized")
         print("="*60)
         print(f"Device         :{self.device}")
         print(f"classes         :{self.num_classes}")
         print(f"Train Batches         :{len(self.train_loader)}")
         print(f"val_batches         :{len(self.val_loader)}")
         print("="*60)

    def train_one_epoch(self) -> Dict[str,float]:
         self.model.train()
         self.train_metrics.reset()

         running_loss = 0.0

         progress_bar = tqdm(
              self.train_loader,
              desc = "Training",
              leave=False,
         )
         for i,batch in enumerate(progress_bar):
              images = batch["pixel_values"].to(self.device)
              labels = batch["labels"].long().to(self.device)

              self.optimizer.zero_grad()

              outputs = self.model(
                   pixel_values = images,
                   labels = labels,
              )
              loss = outputs.loss
              if i==0:
                   print(f"\n first batch loss: {loss.item()}")

              loss.backward()

              self.optimizer.step()

              running_loss += loss.item()

              logits = F.interpolate(
                   outputs.logits,
                   size = labels.shape[-2:],
                   mode="bilinear",
                   align_corners=False,
              )

              predictions = torch.argmax(
                   logits,
                   dim=1,
              )

              self.train_metrics.update(
                   predictions.cpu(),
                   labels.cpu(),
              )

              progress_bar.set_postfix(
                   loss=f"{loss.item(): .4f}"
              )
         epoch_results = self.train_metrics.compute()

         epoch_results["loss"] = (
                running_loss / len(self.train_loader)
              )
         print("\n runnning loss:",running_loss)
         print("no of bathches: ",len(self.train_loader))
         print("AVERAGE LOSS: ", running_loss/len(self.train_loader))

         return epoch_results
    @torch.no_grad()
    def validate_one_epoch(self) -> Dict[str,float]:
         self.model.eval()
         self.val_metrics.reset()
         running_loss = 0.0

         progress_bar = tqdm(
              self.val_loader,
              desc = "validation",
              leave=False,
         )
         for batch in progress_bar:
              images = batch["pixel_values"].to(self.device)
              labels = batch["labels"].long().to(self.device)

              outputs = self.model(
                   pixel_values = images,
                   labels = labels,
              )
              loss = outputs.loss

              running_loss += loss.item()

              logits = F.interpolate(
                   outputs.logits,
                   size = labels.shape[-2:],
                   mode="bilinear",
                   align_corners=False,
              )

              predictions = torch.argmax(
                   logits,
                   dim=1,
              )

              self.val_metrics.update(
                   predictions.cpu(),
                   labels.cpu(),
              )

              progress_bar.set_postfix(
                   loss=f"{loss.item(): .4f}"
              )

         results = self.val_metrics.compute()

         results["loss"] = (
                running_loss / len(self.val_loader)
              )
              
         return results
         
    def save_checkpoint(
              self,
              epoch: int,
              val_miou: float,
    ):
         checkpoint = {
              "epoch": epoch,
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict":self.optimizer.state_dict(),
              "best_miou": val_miou,
         }

         torch.save(
              checkpoint,
              self.checkpoint_dir/"best_model.pth",
         )
         print("\n Best model saved")
    
    def fit(
              self,
              epochs: int,
    ):
         for epoch in range(epochs):
              print("\n" + "=" * 70)
              print(f"Epoch{epoch + 1}/{epochs}")
              print("=" * 70)

              train_results = self.train_one_epoch()
              val_results = self.validate_one_epoch()


              if self.scheduler is not None:
                   self.scheduler.step()
              
              self.history["train_loss"].append(train_results["loss"])
              self.history["val_loss"].append(val_results["loss"])

              self.history["train_miou"].append(train_results["miou"])
              self.history["val_miou"].append(val_results["miou"])

              self.history["train_dice"].append(train_results["dice"])
              self.history["val_dice"].append(val_results["dice"])

              self.history["train_pixel_accuracy"].append(train_results["pixel_accuracy"])
              self.history["val_pixel_accuracy"].append(val_results["pixel_accuracy"])

              print(f"Train Loss : {train_results['loss']:.4f}")
              print(f"val Loss : {val_results['loss']:.4f}")

              print(f"Train miou : {train_results['miou']:.4f}")
              print(f"val miou : {val_results['miou']:.4f}")

              print(f"Train dice : {train_results['dice']:.4f}")
              print(f"val dice : {val_results['dice']:.4f}")

              print(f"Train pixel_accuracy : {train_results['pixel_accuracy']:.4f}")
              print(f"val pixel_accuracy : {val_results['pixel_accuracy']:.4f}")

              if val_results["miou"] > self.best_miou:
                   self.best_miou = val_results["miou"]
                   self.save_checkpoint(
                        epoch,
                        self.best_miou,)
              else:
                   self.counter += 1
                   print(f"no improvement"
                         f"({self.counter}/{self.patience})")
              if self.counter >= self.patience:
                   print("\n Early stopping triggered.")
                   break
              
         return self.history
              