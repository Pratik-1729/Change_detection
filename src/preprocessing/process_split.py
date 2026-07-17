import time
from pathlib import Path
from typing import List
import logging
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from src.preprocessing.process_image import process_image
from src.utils.types import DatasetSummary,ProcessResult,TileMetadata
from src.utils.logging import configure_logging

def process_region(
        dataset_name: str,
        split: str,
        region: str,
        input_root: Path,
        output_root: Path,
)-> List[ProcessResult]:
    
    results = []
    metadata_records = []

    image_dir = input_root/split/region/"images_png"
    mask_dir =  input_root/split/region/"masks_png"

    output_image_dir = output_root/split.lower()/"images"
    output_mask_dir = output_root/split.lower()/"masks"

    image_paths = sorted(image_dir.glob("*.png"))
    logger.info("Processing %s - %s", split, region)

    for image_path in tqdm(image_paths):
        mask_path = mask_dir/image_path.name

        result = process_image(image_path=image_path,
                               mask_path=mask_path,
                               image_output_dir=output_image_dir,
                               mask_output_dir=output_mask_dir,
                               dataset_name=dataset_name,
                               split=split,
                               region=region,
                               )
        results.append(result)
        for tile_index, filename in enumerate(result.saved_images):
            metadata_records.append({
                "filename": filename,
                "split": split,
                "region": region,
                "image_id": image_path.stem.zfill(6),
                "tile": tile_index,
            })
    return results,metadata_records


def process_split(dataset_name: str,
                  split: str,
                  input_root: Path,
                  output_root: Path,)-> DatasetSummary:

    start_time = time.time()

    all_results = []
    all_metadata = []
    for region in ["Urban","Rural"]:
        region_path = input_root/split/region
        if region_path.exists():
            region_results,region_metadata = process_region(dataset_name=dataset_name,
                                            split=split,
                                            region=region,
                                            input_root=input_root,
                                            output_root=output_root,)
            all_results.extend(region_results)
            all_metadata.extend(region_metadata)
    end_time = time.time()
    total_images = len(all_results)
    total_tiles = sum(result.image_tiles for result in all_results)

    summary = DatasetSummary(
        split = split,
        total_images = total_images,
        total_tiles = total_tiles,
        corrupted_images = 0,
        processing_time = end_time - start_time,

    )
    logger.info("%s", "\n" + "=" * 60)
    logger.info("%s SUMMARY", split.upper())
    logger.info("%s", "="*60)
    logger.info("Image Processed: %s", summary.total_images)
    logger.info("Tiles generated: %s", summary.total_tiles)
    logger.info("Processing Time: %.2f seconds", summary.processing_time)
    logger.info("%s", "="*60)

    return summary, all_metadata

if __name__ == "__main__":
    INPUT_ROOT = Path("datasets/LoveDA")
    OUTPUT_ROOT = Path("datasets/processed/LoveDA")

    configure_logging()

    process_split(
        dataset_name="LoveDA",
        split="Train",
        input_root= INPUT_ROOT,
        output_root=OUTPUT_ROOT,
    )