import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from PIL import Image

class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for ISIC2019 federated learning with preprocessing augmentations.
    Loads pre-split client data from .npz files.
    """

    num_classes = 8  # EXCLUDE UNK

    def __init__(self, client_id: int, train: bool = True, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__))

        self.train = train
        self.client_id = client_id

        split = 'train' if train else 'test'
        npz_path = os.path.join(data_path, 'ISIC2019', split, f'{client_id}.npz')

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        client_data = data['data'].item()

        images = client_data['x']   # (N, H, W, C)
        labels = client_data['y']   # (N,)

        # -------------------------
        # REMOVE UNK CLASS (label == 8)
        # -------------------------
        valid_idx = labels != 8
        self.images = images[valid_idx]
        self.labels = labels[valid_idx]

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        sz = 224
        if train:
            self.transforms = albumentations.Compose([
                albumentations.RandomScale(0.07),
                albumentations.Rotate(50),
                albumentations.RandomBrightnessContrast(0.15, 0.1),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Affine(shear=0.1),
                albumentations.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0),
                albumentations.RandomCrop(sz, sz),
                albumentations.CoarseDropout(
                        max_holes=8,
                        max_height=16,
                        max_width=16,
                        fill_value=0,
                        p=0.5
                    ),
            ])
        else:
            self.transforms = albumentations.Compose([
                albumentations.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0),
                albumentations.CenterCrop(sz, sz),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]    # (H, W, 3)
        label = self.labels[idx]    # scalar in [0..7]

        augmented = self.transforms(image=image)
        image = augmented['image']

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
