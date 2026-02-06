import numpy as np
import os
import torch
from collections import defaultdict
import importlib.util

def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
    
    if dataset == 'ISIC2019':
        # Load ISIC2019Dataset dynamically
        module_path = os.path.join(dataset_path, 'isic2019_dataset.py')
        spec = importlib.util.spec_from_file_location("isic2019_dataset", module_path)
        isic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isic_module)
        DatasetClass = isic_module.ISIC2019Dataset
        return DatasetClass(client_id=idx, train=is_train, data_path=dataset_path)
    
    elif dataset == 'ISIC2019_quanv':
        # Load ISIC2019QuanvDataset dynamically
        module_path = os.path.join(dataset_path, 'isic2019_quanv_dataset.py')
        spec = importlib.util.spec_from_file_location("isic2019_quanv_dataset", module_path)
        quanv_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quanv_module)
        DatasetClass = quanv_module.ISIC2019QuanvDataset
        return DatasetClass(client_id=idx, train=is_train, data_path=dataset_path)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def process_image(data, dataset=None):
    X = torch.Tensor(data['x']).type(torch.float32)
    # Transpose from (batch, H, W, C) to (batch, C, H, W) if needed
    if X.dim() == 4 and X.shape[-1] == 3:
        X = X.permute(0, 3, 1, 2)
    
    # Normalize for pretrained models (ImageNet stats)
    if dataset in ['ISIC2019', 'Cifar10', 'Cifar100', 'TinyImagenet', 'Flowers102', 'StanfordCars', 'kvasir', 'Camelyon17', 'iWildCam', 'Country211']:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        X = (X / 255.0 - mean) / std
    
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

