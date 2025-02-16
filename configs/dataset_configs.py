from dataclasses import dataclass


@dataclass
class HAMDatasetConfig:
    seed = 1
    image_shape = (256, 256)
    num_classes = 7

    label_type = "dx"
    image_dir = "data/HAM10000/images"
    mask_dir = "data/HAM10000/masks"
    meta_fpath = "data/HAM10000/metadata.csv"


@dataclass
class ChestXDatasetConfig:
    seed = 1
    image_shape = (256, 256)
    num_classes = 10

    image_dir = "data/ChestX-Det10/images"
    mask_dir = "data/ChestX-Det10/masks"
    meta_fpath = "data/ChestX-Det10/metadata.csv"
