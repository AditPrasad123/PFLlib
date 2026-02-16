"""
Enhanced QuanvEfficientNetB0 with multiple VQC head options for accuracy improvement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from vqc_head_improved import (
    VQCHead, 
    VQCHeadImproved, 
    VQCHeadAdvanced,
    MultiHeadVQCBlock
)


class QuanvEfficientNetB0(nn.Module):
    """
    Original QuanvEfficientNetB0 - for backwards compatibility
    """
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


class QuanvEfficientNetB0Improved(nn.Module):
    """
    Enhanced QuanvEfficientNetB0 with improved VQC head.
    
    Improvements:
    - Uses VQCHeadImproved with 6 qubits instead of 4
    - 3 VQC layers instead of 2
    - Better projection and classification layers
    - Batch normalization for stability
    
    Usage:
        model = QuanvEfficientNetB0Improved(num_classes=8, improvement_level='improved')
    """
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=3, improvement_level='improved'):
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

        # Enhanced VQC head with more qubits and layers
        in_features = backbone.classifier[1].in_features
        if improvement_level == 'advanced':
            self.head = VQCHeadAdvanced(
                in_features=in_features,
                num_classes=num_classes,
                n_qubits=8,
                n_layers=vqc_layers
            )
        else:  # improved (default)
            self.head = VQCHeadImproved(
                in_features=in_features,
                num_classes=num_classes,
                n_qubits=6,
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


class QuanvEfficientNetB0Advanced(nn.Module):
    """
    Most advanced QuanvEfficientNetB0 with state-of-the-art improvements.
    
    Features:
    - 8 qubits for deeper quantum encoding
    - Advanced VQC with parametrized gates
    - Progressive feature reduction
    - Better training dynamics
    
    Usage:
        model = QuanvEfficientNetB0Advanced(num_classes=8)
    """
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=4):
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

        # Most advanced VQC head
        in_features = backbone.classifier[1].in_features
        self.head = VQCHeadAdvanced(
            in_features=in_features,
            num_classes=num_classes,
            n_qubits=8,
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