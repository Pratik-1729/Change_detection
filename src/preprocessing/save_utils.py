from typing import List
import numpy as np
from pathlib import Path
from PIL import Image
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def create_directory(directory:Path)-> None:
     directory.mkdir(parents=True,exist_ok=True)

def save_tiles(tiles:List[np.ndarray],output_dir: Path,dataset_name: str,split:str,region:str,image_id:str,)-> List[str]:
    create_directory(output_dir)
    image_id = str(image_id).zfill(6)
    saved_files = []
    for tile_index,tile in enumerate(tiles):
        filename = (f"{dataset_name}_"f"{split}_"f"{region}_"f"{image_id}_"f"tile_{tile_index}.png")
        save_path = output_dir/filename
        Image.fromarray(tile).save(save_path)
        saved_files.append(filename)
    return saved_files