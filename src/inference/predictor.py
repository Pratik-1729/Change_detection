from pathlib import Path

import logging
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.segformer import build_segformer
from src.utils.config import DEVICE,IMAGE_TILE_SIZE

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self,checkpoint_path: str | Path,):
        self.device = DEVICE

        self.model = build_segformer()

        checkpoint = torch.load(checkpoint_path,map_location=self.device,)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.model.eval()

        self.transform = A.Compose([
            A.Resize(
                IMAGE_TILE_SIZE,
                IMAGE_TILE_SIZE,
            ),
            A.Normalize(),
            ToTensorV2(),
        ])

        logger.info("Predictor Initialized")

    @torch.no_grad()
    def predict(self,image_path: str|Path):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image from path: {image_path}")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB,)
        original_image = image.copy()
        transformed = self.transform(image=image,)

        image_tensor = transformed["image"]
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        outputs = self.model(
            pixel_values = image_tensor,
        )

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(IMAGE_TILE_SIZE,IMAGE_TILE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        prediction = torch.argmax(logits,dim=1,)

        prediction = prediction.squeeze(0)
        prediction = prediction.cpu().numpy()
        probabilities = torch.softmax(logits,dim=1)
        confidence = probabilities.max(dim=1).values

        return (
            original_image,
            prediction,
            confidence.cpu().numpy())
