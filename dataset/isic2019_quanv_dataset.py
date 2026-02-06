import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from PIL import Image

class ISIC2019QuanvDataset(Dataset):
    num_classes = 8

    def __init__(self, client_id: int, train: bool = True, data_path: str = None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')

        self.train = train
        self.client_id = client_id

        split = 'train' if train else 'test'
        npz_path = os.path.join(data_path, 'ISIC2019', split, f'{client_id}.npz')

        data = np.load(npz_path, allow_pickle=True)
        client_data = data['data'].item()

        self.images = client_data['x']   # (N, H, W, 4)
        self.labels = client_data['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if y < 0 or y >= 8:
            raise ValueError(f"Invalid label: {y}")

        # Remove accidental extra dimension (H, W, C, 1)
        if isinstance(x, np.ndarray) and x.ndim == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 3:
            # (H, W, C) → (C, H, W)
            x = x.permute(2, 0, 1)
        elif x.ndim == 4:
            # (1, C, H, W)
            x = x.squeeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")

        assert x.shape[0] == 4, f"Expected 4 channels, got {x.shape}"

        return x, torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    # Test the dataset
    dataset = ISIC2019QuanvDataset(client_id=0, train=True)
    print(f"Train dataset size: {len(dataset)}")
    print(f"Sample image shape: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")

    dataset_test = ISIC2019QuanvDataset(client_id=0, train=False)
    print(f"Test dataset size: {len(dataset_test)}")
    print(f"Sample test image shape: {dataset_test[0][0].shape}")