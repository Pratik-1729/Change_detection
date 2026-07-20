from pathlib import Path
import cv2
import pandas as pd
from torch.utils.data import Dataset
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
class LoveDADataset(Dataset):
    def __init__(self,
                 processed_root: Path,
                 split:str,
                 transform: None,):
        self.processed_root = Path(processed_root)
        self.split = split
        if self.split.lower() == "train":
            self.split = "Train"
        elif self.split.lower() == "val":
            self.split = "Val"
        self.transform = transform
        metadata = pd.read_csv(self.processed_root/"metadata.csv")
        self.metadata = metadata[metadata["split"]==split].reset_index(drop=True)
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        image_path = (
            self.processed_root/self.split/"images"/row["filename"]
        )

        mask_path = (self.processed_root/self.split/"masks"/row["filename"])

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            str(mask_path),cv2.IMREAD_GRAYSCALE
        )

        if self.transform:
            transformed = self.transform(image=image,
                                         mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return {"pixel_values": image,
                "labels": mask.long(),
                "filename": row["filename"],
                }