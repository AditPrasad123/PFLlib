import numpy as np
import os
import torch
from collections import defaultdict
from dataset.isic2019_dataset import ISIC2019Dataset
from dataset.isic2019_quanv_dataset import ISIC2019QuanvDataset

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
        return ISIC2019Dataset(client_id=idx, train=is_train, data_path=dataset_path)
    else:
        return ISIC2019QuanvDataset(client_id=idx, train=is_train, data_path=dataset_path, few_shot=few_shot)

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

