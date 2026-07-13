from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
import numpy as np

from src.data.class_mapping import remap_mask
from src.utils.config import LOVEDA_DIR
VALID_CLASSES = {0,1,2,3,4}
def verify_folder(folder_path : Path) -> dict[str,int]:
    image_folder = folder_path / "images_png"
    mask_folder = folder_path / "masks_png"
    image_files = sorted(list(image_folder.glob("*.png")))
    mask_files = sorted(list(mask_folder.glob("*.png")))
    
    print(f"\n checking Folder : {folder_path.name}")
    print("="* 60)

    print(f"Images: {len(image_files)}")
    print(f"masks: {len(mask_files)}")

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
            print(f"Cannot open : {image_path.name}")
            print(e)

            corrupted += 1
            continue

        if image.shape[:2] != mask.shape[:2]:
            print(f"Shape Mismatch : {image_path.name}")
            mismatched+=1

            remapped_mask = remap_mask(mask)
            labels = set(np.unique(remapped_mask))

            if not labels.issubset(VALID_CLASSES):
                print(f"Invalid labels found in {mask_path.name}")
    print()

    print("Folder Verification Complete")
    print(f"corrupted files: {corrupted}")
    print(f"Mismatched Files: {mismatched}")
    print()
    return{
        "images" : len(image_files),
        "corrupted" : corrupted,
        "mismatched": mismatched
    }

def verify_split(split_name:str) -> None:
    
    print("\n")
    print("="*60)
    print(f"{split_name.upper()} DATASET")
    print("="*60)
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
    print("="*60)
    print(f"{split_name.upper()} SUMMARY")
    print("="*60)
    print(f"Total_Images      : {total_images}")
    print(f"Total_corrupted      : {total_corrupted}")
    print(f"Total_mismatched      : {total_mismatched}")
    print("="*60)

def main() -> None:
    verify_split("Train")
    # verify_split("Test")
    verify_split("Val")

if __name__ == "__main__":
    main()

    