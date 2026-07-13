from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.types import TileMetadata

tile= TileMetadata(
    dataset='LoveDA',
    split='Train',
    region='Urban',
    image_id='00001',
    tile_index=0,
    filename="LoveDA_Train_Urban_00001_tile_0.png"
)

print(tile)