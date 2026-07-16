import sys
from pathlib import Path
from transformers import SegformerForSemanticSegmentation 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import (
    MODEL_NAME,NUM_CLASSES
)
def build_segformer(
        num_classes: int = 5,
):
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model