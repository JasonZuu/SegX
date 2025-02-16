import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import PIL

from configs.dataset_configs import HAMDatasetConfig
from utils.misc import random_split_dataset, category_to_onehot
from utils.torch_transforms import get_image_transform, get_mask_transform


ham_label_to_idx = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}

class HAMDataset(Dataset):
    def __init__(self, config: HAMDatasetConfig,
                 set_type='train', load_mask=False, return_image_name=False):
        self.config = config
        self.image_dir = self.config.image_dir
        self.mask_dir = self.config.mask_dir
        self.meta_fpath = self.config.meta_fpath
        self.set_type = set_type
        self.num_classes = len(ham_label_to_idx)
        self.return_image_name = return_image_name
        self.load_mask = load_mask
        
        self.image_transform = get_image_transform(self.config.image_shape)
        self.mask_transform = get_mask_transform(self.config.image_shape)
        self.label_type = self.config.label_type

        self.image_names, self.image_paths, self.mask_paths, self.labels = self._load_data()

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = self.image_paths[index]
        image = PIL.Image.open(img_path, mode='r').convert('RGB')

        image = self.image_transform(image)
        y = self.labels[index]
            
        if self.load_mask:
            mask_path = self.mask_paths[index]
            mask = PIL.Image.open(mask_path, mode='r').convert('L')
            mask = self.mask_transform(mask)
        
        if self.return_image_name and self.load_mask:
            return image, mask, y, image_name
        elif self.return_image_name:
            return image, y, image_name
        elif self.load_mask:
            return image, mask, y
        else:
            return image, y

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        meta_data = pd.read_csv(self.meta_fpath,
                                usecols=['image_id', "dx"],
                                encoding="utf-8")
        image_names = meta_data["image_id"].values

        image_paths = np.array([os.path.join(self.image_dir, f"{image_name}.jpg") for image_name in image_names])
        mask_paths = np.array([os.path.join(self.mask_dir, f"{image_name}_segmentation.png") for image_name in image_names])
        labels = meta_data["dx"].values
        labels = np.array([ham_label_to_idx[label] for label in labels])
        labels = category_to_onehot(labels, num_classes=self.num_classes)

        image_names, image_paths, mask_paths, labels = random_split_dataset(image_names, image_paths, mask_paths, labels,
                                                                            set_type=self.set_type, seed=self.config.seed)
        return image_names, image_paths, mask_paths, labels
    
    def get_labels(self):
        return self.labels
    
    def get_image_shape(self):
        return self.config.image_shape
    
    