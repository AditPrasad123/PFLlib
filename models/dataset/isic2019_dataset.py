import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from PIL import Image

class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for ISIC2019 federated learning
    using quanvolved features.
    """

    num_classes = 8

    def __init__(self, client_id: int, train: bool = True, data_path: str = None):
        if data_path is None:
            data_path = "D:\\Capstone\\CodeBase\\PFLlib\\dataset"

        self.train = train
        self.client_id = client_id

        split = 'train_quanv' if train else 'test_quanv'
        npz_path = os.path.join(data_path, 'ISIC2019', split, f'{client_id}.npz')

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        client_data = data['data'].item()

        self.images = client_data['x']  # (N, 12, 12, 1)
        self.labels = client_data['y']  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]          # (12,12,1)
        label = self.labels[idx]

        # Convert to torch tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)      # → (1,12,12)

        label = torch.tensor(label, dtype=torch.long)

        return img, label



if __name__ == "__main__":
    # Test the dataset
    dataset = ISIC2019Dataset(client_id=0, train=True)
    print(f"Train dataset size: {len(dataset)}")
    print(f"Sample image shape: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")

    dataset_test = ISIC2019Dataset(client_id=0, train=False)
    print(f"Test dataset size: {len(dataset_test)}")
    print(f"Sample test image shape: {dataset_test[0][0].shape}")