import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from vqc_head import VQCHead

class QuanvTinyViT(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=2):
        super().__init__()

        # Adapter: 4 → 3
        self.adapter = nn.Conv2d(4, 3, kernel_size=1)

        # Create TinyViT WITHOUT classifier
        backbone = create_model(
            "tiny_vit_5m_224.dist_in22k",
            pretrained=pretrained,
            num_classes=0   # IMPORTANT: no head inside
        )

        # 🔑 FedBABU: shared BODY
        self.base = nn.Sequential(
            backbone,
            nn.Identity()   # keeps interface simple
        )

        # 🔑 FedBABU: personalized HEAD (VQC)
        self.head = VQCHead(
            in_features=backbone.num_features,
            num_classes=num_classes,
            n_qubits=4,
            n_layers=vqc_layers
        )
        self.fc = self.head

    def forward(self, x):
        # x: (B, 4, H, W)
        x = self.adapter(x)

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        features = self.base(x)   # (B, D)
        out = self.head(features)
        return out