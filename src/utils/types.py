from dataclasses import dataclass
from typing import List

@dataclass
class TileMetadata:
    dataset: str
    split: str
    region: str
    image_id: str
    tile_index: int
    filename: str


@dataclass
class ProcessResult:
    image_name : str
    image_tiles : int
    mask_tiles: int
    saved_images: List[str]
    saved_masks: List[str]
    split: str
    region: str
    # metadata: List[TileMetadata]

@dataclass
class DatasetSummary:
    split: str
    total_images: int
    total_tiles: int
    corrupted_images: int
    processing_time: float

@dataclass
class TrainingResult:
    epoch: int
    train_loss: float
    validation_loss: float
    mIoU: float
    f1_score: float
    learning_rate: float

@dataclass
class PredictionResult:
    filename: str
    prediction_path: str
    inference_time: float

@dataclass
class ChangeStatistics:
    building_before: float
    building_after: float
    road_before: float
    road_after: float
    vegetation_before: float
    vegetation_after: float
    water_before: float
    water_after: float
    new_construction_area: float
    demolished_area: float
