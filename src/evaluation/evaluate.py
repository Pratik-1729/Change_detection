from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.segformer import build_segformer
from src.data.dataloader import get_dataloaders
from src.evaluation.metrics import SegmentationMetrics
from src.utils.constants import CLASS_NAMES
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.report import(
    save_csv,
    save_json,
    save_summary
)

from src.utils.config import (
    DEVICE,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    PROCESSED_DATASET,
    IMAGE_TILE_SIZE,
)
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self,
                 checkpoint_path:Path):
        self.device = DEVICE
        self.metrics = SegmentationMetrics(NUM_CLASSES)
        _, self.val_loader = get_dataloaders(
            processed_root=PROCESSED_DATASET / "LoveDA"
        )

        self.model = build_segformer()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_path = checkpoint_path
        logger.info("%s", "=" *60)
        logger.info("Evaluator Initialized")
        logger.info("%s", "=" *60)
        logger.info("Checkpoint  :%s", checkpoint_path.name)
        logger.info("Device      :%s", self.device)
        logger.info("Images      :%s", len(self.val_loader.dataset))
        logger.info("%s", "=" *60)
    @torch.no_grad()
    def evaluate(self):
        self.metrics.reset()
        running_loss = 0.0

        progress = tqdm(
            self.val_loader,
            desc="Evaluating",
        )

        for batch in progress:
            images = batch["pixel_values"].to(self.device)
            labels = batch["labels"].long().to(self.device)

            outputs = self.model(pixel_values = images,labels = labels,)

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

            self.metrics.update(
                predictions.cpu(),
                labels.cpu(),
              )

            progress.set_postfix(
                loss=f"{loss.item(): .4f}"
            )

            
        results = self.metrics.compute()
            
        results["loss"] = (
            running_loss / len(self.val_loader)
        )

        return results
    
def print_results(results):
    logger.info("%s", "\n"+ "=" *60)
    logger.info("OVERALL METRICS")
    logger.info("%s", "=" *60)
    logger.info("Loss                :%.4f", results['loss'])
    logger.info("pixel_accuracy      :%.4f", results['pixel_accuracy'])
    logger.info("Precision           :%.4f", results['precision'])
    logger.info("Recall              :%.4f", results['recall'])
    logger.info("F1 Score            :%.4f", results['f1_score'])
    logger.info("Mean IoU            :%.4f", results['miou'])
    logger.info("Dice Score            :%.4f", results['dice'])

    logger.info("%s", "\n"+ "=" *60)
    logger.info("PER-CLASS METRICS")
    logger.info("%s", "=" *60)

    for class_id,class_name in CLASS_NAMES.items():
        logger.info("%s", f"\n {class_name}")
        logger.info("IoU            :%s", results['per_class_iou'][class_id])
        logger.info("Dice           :%s", results['per_class_dice'][class_id])
        logger.info("Precision      :%s", results['per_class_precision'][class_id])
        logger.info("Recall         :%s", results['per_class_recall'][class_id])
        logger.info("Support        :%s", int(results['support'][class_id]))


def main():

    evaluator = Evaluator(
        checkpoint_path=CHECKPOINT_DIR/"best_model.pth"
    )
    results = evaluator.evaluate()
    save_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        output_dir=Path("outputs/evaluation"),
        normalize=False,
    )
    save_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        output_dir=Path("outputs/evaluation"),
        normalize=True,
    )
    print_results(results)
    output_dir = Path("outputs/evaluation")
    save_json(results,output_dir)
    save_csv(results,output_dir)
    save_summary(results,output_dir)
if __name__ == "__main__":
    main()