import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.class_mapping import remap_mask

sample = np.array([
    [0,1,2],
    [3,4,5],
    [6,5,4]
])

result = remap_mask(sample)
print(result)