"""
Enhanced QuanvTinyViT with multiple VQC head options for accuracy improvement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from vqc_head_improved import (
    VQCHead, 
    VQCHeadImproved, 
    VQCHeadAdvanced,
    MultiHeadVQCBlock
)


class QuanvTinyViT(nn.Module):
    """
    Original QuanvTinyViT - for backwards compatibility
    """
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=2):
        super().__init__()

        # Adapter: 4 → 3
        self.adapter = nn.Conv2d(4, 3, kernel_size=1)

        # Create TinyViT WITHOUT classifier
        self.base = create_model(
            "tiny_vit_5m_224.dist_in22k",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )

        # 🔑 FedBABU: personalized HEAD (VQC)
        self.head = VQCHead(
            in_features=self.base.num_features,
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


class QuanvTinyViTImproved(nn.Module):
    """
    Enhanced QuanvTinyViT with improved VQC head.
    
    Improvements:
    - Uses VQCHeadImproved with 6 qubits instead of 4
    - 3 VQC layers instead of 2
    - Better projection and classification layers
    - Batch normalization for stability
    
    Usage:
        model = QuanvTinyViTImproved(num_classes=8, improvement_level='improved')
    """
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=3, improvement_level='improved'):
        super().__init__()

        # Adapter: 4 → 3
        self.adapter = nn.Conv2d(4, 3, kernel_size=1)

        # Create TinyViT WITHOUT classifier
        self.base = create_model(
            "tiny_vit_5m_224.dist_in22k",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )

        # Enhanced VQC head with more qubits and layers
        if improvement_level == 'advanced':
            self.head = VQCHeadAdvanced(
                in_features=self.base.num_features,
                num_classes=num_classes,
                n_qubits=8,
                n_layers=vqc_layers
            )
        else:  # improved (default)
            self.head = VQCHeadImproved(
                in_features=self.base.num_features,
                num_classes=num_classes,
                n_qubits=6,
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


class QuanvTinyViTAdvanced(nn.Module):
    """
    Most advanced QuanvTinyViT with state-of-the-art improvements.
    
    Features:
    - 8 qubits for deeper quantum encoding
    - Advanced VQC with parametrized gates
    - Progressive feature reduction
    - Better training dynamics
    
    Usage:
        model = QuanvTinyViTAdvanced(num_classes=8)
    """
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=4):
        super().__init__()

        # Adapter: 4 → 3
        self.adapter = nn.Conv2d(4, 3, kernel_size=1)

        # Create TinyViT WITHOUT classifier
        self.base = create_model(
            "tiny_vit_5m_224.dist_in22k",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )

        # Most advanced VQC head
        self.head = VQCHeadAdvanced(
            in_features=self.base.num_features,
            num_classes=num_classes,
            n_qubits=8,
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