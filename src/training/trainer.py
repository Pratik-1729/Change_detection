from pathlib import Path
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.evaluation.metrics import SegmentationMetrics
from src.evaluation.plots import plot_training_history
from src.utils.config import OUTPUT_DIR
from src.utils.config import(
    CHECKPOINT_DIR,
    NUM_CLASSES,
)
import logging

logger = logging.getLogger(__name__)

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
         logger.info("%s", "="*60)
         logger.info("Trainer Initialized")
         logger.info("%s", "="*60)
         logger.info("Device         :%s", self.device)
         logger.info("classes         :%s", self.num_classes)
         logger.info("Train Batches         :%s", len(self.train_loader))
         logger.info("val_batches         :%s", len(self.val_loader))
         logger.info("%s", "="*60)

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
                   logger.debug("first batch loss: %s", loss.item())

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
         logger.info("running loss: %s", running_loss)
         logger.info("no of batches: %s", len(self.train_loader))
         logger.info("AVERAGE LOSS: %s", running_loss/len(self.train_loader))

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
              filename: str,
    ):
         checkpoint = {
              "epoch": epoch,
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict":self.optimizer.state_dict(),
              "scheduler_state_dict":(
               self.scheduler.state_dict()
               if self.self.scheduler is not None
               else None
              ),
              "best_miou": val_miou,
              "history" : self.history,
         }
         save_path = self.checkpoint_dir/filename
         torch.save(
              checkpoint,
              save_path,
         )
         logger.info("Best model saved to %s", str(save_path)) 

    def load_checkpoint(
              self,
              filename: str = "last_model.pth"
    ):
         checkpoint_path = self.checkpoint_dir/ filename
         if not checkpoint_path.exists():
              logger.info(f"no checkpoint found at{checkpoint_path}")
              return 0
         checkpoint = torch.load(
              checkpoint_path,
              map_location=self.device,
         )

         self.model.load_state_dict(
              checkpoint["model_state_dict"]
         )

         self.optimizer.load_state_dict(
              checkpoint["optimizer_state_dict"]
         )

         if(
              self.scheduler is not None 
              and checkpoint["scheduler_state_dict"] is not None
         ):
              self.scheduler.load_state_dict(
              checkpoint["scheduler_state_dict"])
         
         self.best_miou = checkpoint["best_miou"]
         self.history = checkpoint["history"]

         start_epoch = checkpoint["epoch"] + 1

         logger.info("="*60)
         logger.info("Checkpoint loaded Successfully")
         logger.info(f"Resuming from epoch{start_epoch}")
         logger.info(f"Best Miou : {self.best_miou:.4f}")
         logger.info("="*60)

         return start_epoch
          
              
    def fit(
              self,
              epochs: int,
              resume: bool =  False,
    ):
         start_epoch = 0
         if resume:
              start_epoch = self.load_checkpoint()

         for epoch in range(start_epoch,epochs):
              logger.info("%s", "\n" + "=" * 70)
              logger.info("Epoch %s/%s", epoch + 1, epochs)
              logger.info("%s", "=" * 70)

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

              logger.info("Train Loss : %.4f", train_results['loss'])
              logger.info("val Loss : %.4f", val_results['loss'])

              logger.info("Train miou : %.4f", train_results['miou'])
              logger.info("val miou : %.4f", val_results['miou'])

              logger.info("Train dice : %.4f", train_results['dice'])
              logger.info("val dice : %.4f", val_results['dice'])

              logger.info("Train pixel_accuracy : %.4f", train_results['pixel_accuracy'])
              logger.info("val pixel_accuracy : %.4f", val_results['pixel_accuracy'])

              self.save_checkpoint(
                   epoch=epoch,
                   val_miou=val_results["miou"],
                   filename="last_model.pth",
              )

              if val_results["miou"] > self.best_miou:
                   self.best_miou = val_results["miou"]
                   self.counter = 0
                   self.save_checkpoint(
                        epoch = epoch,
                        val_miou=self.best_miou,
                        filename="best_model.pth",
                   )
              else:
                   self.counter += 1
                   logger.info("no improvement (%s/%s)", self.counter, self.patience)
              if self.counter >= self.patience:
                   logger.info("Early stopping triggered.")
                   break
         plot_training_history(
              history=self.history,
              output_dir=OUTPUT_DIR/"training",
         )
              
         return self.history
              