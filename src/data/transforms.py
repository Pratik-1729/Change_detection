import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.Resize(512,512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=    0.2,
                                   contrast_limit=0.2,p=0.2),
        A.Normalize(
            mean=(0.485,0.456,0.406),
            std = (0.229,0.224,0.225)
        ),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(512,512),
        A.Normalize(
            mean=(0.485,0.456,0.406),
            std = (0.229,0.224,0.225)
        ),
        ToTensorV2()
    ])
