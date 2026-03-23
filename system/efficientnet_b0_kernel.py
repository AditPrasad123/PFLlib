"""
EfficientNetB0 backbone with compact embedding projection for kernel-based heads.

Designed for FedBABU:
- self.base is globally shared/aggregated
- self.head is locally personalized
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0KernelFedBABU(nn.Module):
    def __init__(
        self,
        num_classes=9,
        pretrained=True,
        projection_dim=128,
        embedding_dim=8,
    ):
        super().__init__()

        backbone = efficientnet_b0(pretrained=pretrained)
        in_features = backbone.classifier[1].in_features

        # FedBABU shared body: feature extractor + compact embedding projector.
        self.base = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
            nn.Linear(in_features, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(projection_dim, embedding_dim),
        )

        # FedBABU personalized head (differentiable during federated rounds).
        self.head = nn.Linear(embedding_dim, num_classes)

        # Keep compatibility with existing FL framework helpers.
        self.fc = self.head
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def forward(self, x):
        emb = self.base(x)
        return self.head(emb)

    def extract_embeddings(self, x):
        return self.base(x)

    def forward_from_embeddings(self, emb):
        return self.head(emb)
