from pathlib import Path

LOVEDA_CONFIG ={
    "name":"LoveDA",
    "root" : Path("datasets/LoveDA"),
    "processed_root": Path("datasets/processed/LoveDA"),
    "splits":[
        "Train",
        "Val"
    ],
    "regions":[
        "Urban",
        "Rural",
    ],
    "image_folder":"images_png",
    "masked_folder":"masks_png",
    "tile_size":512,
    "num_classes":5
}