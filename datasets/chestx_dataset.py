import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import PIL

from configs.dataset_configs import ChestXDatasetConfig
from utils.misc import random_split_dataset, category_to_onehot
from utils.torch_transforms import get_image_transform, get_mask_transform


chestx_label_to_idx = {
    'Atelectasis': 0,
    'Calcification': 1,
    'Consolidation': 2,
    'Effusion': 3,
    'Emphysema': 4,
    'Fibrosis': 5,
    'Fracture': 6,
    'Mass': 7,
    'Nodule': 8,
    'Pneumothorax': 9
}


class ChestXDataset(Dataset):
    def __init__(self, config: ChestXDatasetConfig, set_type='train',
                 load_mask=False, return_image_name=False):
        self.config = config
        self.image_dir = self.config.image_dir
        self.mask_dir = self.config.mask_dir
        self.meta_fpath = self.config.meta_fpath
        self.set_type = set_type
        self.num_classes = len(chestx_label_to_idx)

        self.load_mask = load_mask
        self.return_image_name = return_image_name
        
        self.image_transform = get_image_transform(self.config.image_shape)
        self.mask_transform = get_mask_transform(self.config.image_shape)

        # Load image paths, mask paths, and labels
        self.image_names, self.image_paths, self.mask_paths, self.labels = self._load_data()

    def __getitem__(self, index):
        # Load the image
        img_path = self.image_paths[index]
        image = PIL.Image.open(img_path, mode='r').convert('RGB')
        image = self.image_transform(image)

        # Load the label
        y = torch.tensor(self.labels[index])

        # Load corresponding masks (if load_mask is True)
        if self.return_image_name and self.load_mask:
            image_name = self.image_names[index]
            mask = self._load_mask_fn(index)
            return image, mask, y, image_name
        elif self.load_mask:
            mask = self._load_mask_fn(index)
            return image, mask, y
        elif self.return_image_name:
            image_name = self.image_names[index]
            return image, y, image_name
        else:
            return image, y

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        # Load metadata
        meta_data = pd.read_csv(self.meta_fpath, encoding="utf-8")
        image_names = meta_data["image_id"].values

        # Create image paths
        image_paths = np.array([os.path.join(self.image_dir, f"{image_name}.png") for image_name in image_names])
        mask_paths = np.array([os.path.join(self.mask_dir, f"{image_name}.png") for image_name in image_names])
        
        # Extract labels based on disease columns
        labels = np.zeros((len(image_names), self.num_classes), dtype=np.float32)
        for disease, idx in chestx_label_to_idx.items():
            labels[:, idx] = meta_data[disease].values

        # Optionally split dataset into train/validation/test sets
        image_names, image_paths, mask_paths, labels = random_split_dataset(image_names, image_paths, mask_paths, labels,
                                                                                 set_type=self.set_type, seed=self.config.seed)
        return image_names, image_paths, mask_paths, labels
    
    def get_labels(self):
        return self.labels
    
    def get_image_shape(self):
        return self.config.image_shape
    
    def _load_mask_fn(self, index):
        mask_path = self.mask_paths[index]
        mask = PIL.Image.open(mask_path, mode='r').convert('L')
        mask = self.mask_transform(mask)
        return mask


if __name__ == "__main__":
    config = ChestXDatasetConfig()
    dataset = ChestXDataset(config, set_type='val', load_mask=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        images, masks, labels = data
        print(images.shape, masks.shape, labels.shape)
        break
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Image shape: {dataset.get_image_shape()}")
    print(f"Labels: {dataset.get_labels()}")
