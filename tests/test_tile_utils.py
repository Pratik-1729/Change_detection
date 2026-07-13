import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.tiles_utils import split_into_tiles

image = np.zeros((1024,1024,3),dtype=np.uint8)
tiles = split_into_tiles(image)

print(f"Number of tiles : {len(tiles)}")

for i,tile in enumerate(tiles):
    print(f"tile {i} : {tile.shape}")
