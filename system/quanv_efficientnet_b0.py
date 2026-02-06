import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from vqc_head import VQCHead

class QuanvEfficientNetB0(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=2):
        super().__init__()

        # Adapter: 4 → 3
        self.adapter = nn.Conv2d(4, 3, kernel_size=1)

        # EfficientNet backbone
        backbone = efficientnet_b0(pretrained=pretrained)

        # 🔑 FedBABU: define BASE (shared)
        self.base = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten()
        )

        # 🔑 FedBABU: define HEAD (personalized, VQC)
        in_features = backbone.classifier[1].in_features
        self.head = VQCHead(
            in_features=in_features,
            num_classes=num_classes,
            n_qubits=4,
            n_layers=vqc_layers
        )
        self.fc = self.head

    def forward(self, x):
        # x: (B, 4, 24, 24)

        x = self.adapter(x)

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        features = self.base(x)
        out = self.head(features)
        return out