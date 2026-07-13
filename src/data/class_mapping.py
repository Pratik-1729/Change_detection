import numpy as np
LOVEDA_CLASS_MAPPING = {
    0 : 0,
    1 : 1,
    2 : 2,
    3 : 4,
    4 : 0,
    5 : 3,
    6 : 3
}

def remap_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(mask)
    for old_class, new_class in LOVEDA_CLASS_MAPPING.items():
        remapped[mask == old_class] = new_class
    return remapped