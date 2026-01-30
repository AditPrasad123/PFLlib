from dataset.isic2019_dataset import ISIC2019Dataset
import numpy as np

dataset = ISIC2019Dataset(client_id=0, train=False)
x, y = dataset[0]

print(x.shape)   # should be (1, 12, 12)
print(x.min(), x.max())

