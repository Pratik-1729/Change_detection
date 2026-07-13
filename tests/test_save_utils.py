import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.save_utils import save_tiles

tiles = []

for _ in range(4):
    tile = np.random.randint(0,255,(512,512,3),dtype=np.uint8)
    tiles.append(tile)

saved_files = save_tiles(tiles=tiles,output_dir=Path("temp_tiles"),dataset_name="LoveDA",split="Train",region="Urban",image_id="1")
print("\n saved files")

for files in saved_files:
    print(files)