from typing import List
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def split_into_tiles(image:np.ndarray, tile_size: int = 512) -> List[np.ndarray]:
    height,width = image.shape[:2]
    if height % tile_size !=0 or width % tile_size != 0:
        raise ValueError(
            f"Image size{height},{width} is not divisible by tile size({tile_size})"

        )
    tiles = []

    for y in range(0,height,tile_size):
        for x in range(0,width,tile_size):
            tile = image[y: y+tile_size,
                         x : x+ tile_size
                         ]
            tiles.append(tile)
    return tiles