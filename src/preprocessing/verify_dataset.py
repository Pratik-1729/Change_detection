from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

from src.data.class_mapping import remap_mask
from src.utils.config import LOVEDA_DIR
VALID_CLASSES = {0,1,2,3,4}
def verify_folder(folder_path : Path) -> dict[str,int]:
    image_folder = folder_path / "images_png"
    mask_folder = folder_path / "masks_png"
    image_files = sorted(list(image_folder.glob("*.png")))
    mask_files = sorted(list(mask_folder.glob("*.png")))
    
    logger.info("Checking folder: %s", folder_path.name)
    logger.info("%s", "="* 60)

    logger.info("Images: %s", len(image_files))
    logger.info("Masks: %s", len(mask_files))

    if len(image_files) != len(mask_files) :
        raise ValueError(
            f"Number of images and masks do not match in {folder_path.name}"
        )
    corrupted = 0
    mismatched = 0

    for image_path, mask_path in zip(image_files,mask_files):
        try:
            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

        except Exception as e:
            logger.exception("Cannot open: %s", image_path.name)

            corrupted += 1
            continue

        if image.shape[:2] != mask.shape[:2]:
            logger.warning("Shape mismatch: %s", image_path.name)
            mismatched += 1

            remapped_mask = remap_mask(mask)
            labels = set(np.unique(remapped_mask))

            if not labels.issubset(VALID_CLASSES):
                logger.warning("Invalid labels found in %s", mask_path.name)
    logger.info("")

    logger.info("Folder Verification Complete")
    logger.info("corrupted files: %s", corrupted)
    logger.info("Mismatched Files: %s", mismatched)
    logger.info("")
    return{
        "images" : len(image_files),
        "corrupted" : corrupted,
        "mismatched": mismatched
    }

def verify_split(split_name:str) -> None:
    
    logger.info("\n")
    logger.info("%s", "="*60)
    logger.info("%s DATASET", split_name.upper())
    logger.info("%s", "="*60)
    split_path = LOVEDA_DIR/ split_name

    total_images = 0
    total_corrupted = 0
    total_mismatched = 0
    
    for folder in ["Urban","Rural"]:
        folder_path = split_path/folder

        result = verify_folder(folder_path)

        total_images += result["images"]
        total_corrupted += result["corrupted"]
        total_mismatched += result["mismatched"]
    logger.info("%s", "="*60)
    logger.info("%s SUMMARY", split_name.upper())
    logger.info("%s", "="*60)
    logger.info("Total_Images      : %s", total_images)
    logger.info("Total_corrupted   : %s", total_corrupted)
    logger.info("Total_mismatched  : %s", total_mismatched)
    logger.info("%s", "="*60)

def main() -> None:
    from src.utils.logging import configure_logging
    configure_logging()

    verify_split("Train")
    # verify_split("Test")
    verify_split("Val")

if __name__ == "__main__":
    main()

    