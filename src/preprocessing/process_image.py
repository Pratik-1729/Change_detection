import numpy as np
from pathlib import Path
from PIL import Image

from src.data.class_mapping import remap_mask
from src.preprocessing.save_utils import save_tiles
from src.preprocessing.tiles_utils import split_into_tiles
from src.utils.types import ProcessResult

def process_image(
        image_path: Path,
        mask_path: Path,
        image_output_dir: Path,
        mask_output_dir: Path,
        dataset_name: str,
        split: str,
        region: str,
)-> ProcessResult:
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))

    mask = remap_mask(mask)

    image_tiles = split_into_tiles(image)
    mask_tiles = split_into_tiles(mask)

    image_id = image_path.stem


    saved_images = save_tiles(
        tiles=image_tiles,
        output_dir=image_output_dir,
        dataset_name=dataset_name,
        split=split,
        region=region,
        image_id=image_id,
    )

    saved_masks = save_tiles(
        tiles=mask_tiles,
        output_dir=mask_output_dir,
        dataset_name=dataset_name,
        split=split,
        region=region,
        image_id=image_id,
    )


    return ProcessResult(
        image_name=image_path.name,
        image_tiles=len(image_tiles),
        mask_tiles=len(mask_tiles),
        saved_images=saved_images,
        saved_masks=saved_masks,
        split=split,
        region=region,
    )