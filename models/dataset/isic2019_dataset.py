import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from PIL import Image

class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for ISIC2019 federated learning with preprocessing augmentations.

    Loads pre-split client data from .npz files and applies the same augmentations
    as the original FedIsic2019 dataset.
    """
    
    num_classes = 9  # MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK

    def __init__(self, client_id: int, train: bool = True, data_path: str = None):
        """
        Args:
            client_id (int): Client ID (0-5)
            train (bool): Whether to load train or test data
            data_path (str): Path to the dataset directory. If None, uses default.
        """
        if data_path is None:
            data_path = "D:\\Capstone\\CodeBase\\PFLlib\\dataset"

        self.train = train
        self.client_id = client_id

        # Load the .npz file
        split = 'train' if train else 'test'
        npz_path = os.path.join(data_path, 'ISIC2019', split, f'{client_id}.npz')

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        # Load with allow_pickle=True since it contains dict
        data = np.load(npz_path, allow_pickle=True)
        client_data = data['data'].item()

        self.images = client_data['x']  # Shape: (N, H, W, C) - numpy arrays
        self.labels = client_data['y']  # Shape: (N,) - class indices

        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Define augmentations - images are 224px height with varying width
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
            # Test: apply center crop to get consistent 224x224 output
            self.transforms = albumentations.Compose([
                albumentations.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0),
                albumentations.CenterCrop(sz, sz),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (H, W, 3) - varying dimensions
        label = self.labels[idx] # Convert to 0-based index (0-8), removing UNK class
        if label < 0 or label >= 8:
            raise ValueError(f"Invalid label after shift: {label}")

        # Apply augmentations (train: random augments + crop, test: center crop)
        # PadIfNeeded ensures image is at least 224x224 before cropping
        augmented = self.transforms(image=image)
        image = augmented['image']  # Now 224x224x3
        
        # Normalize: convert to float, scale to [0,1], then apply ImageNet stats
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # Test the dataset
    dataset = ISIC2019Dataset(client_id=0, train=True)
    print(f"Train dataset size: {len(dataset)}")
    print(f"Sample image shape: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")

    dataset_test = ISIC2019Dataset(client_id=0, train=False)
    print(f"Test dataset size: {len(dataset_test)}")
    print(f"Sample test image shape: {dataset_test[0][0].shape}")