"""
Enhanced QuanvTinyViT with multiple head options: VQC (quantum) or Standard (classical).
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


class StandardClassifierHead(nn.Module):
    """
    Standard classical neural network classifier head.
    
    Architecture:
    - Projection layer: in_features → 256
    - Hidden layer: 256 → 128 with ReLU and BatchNorm
    - Output layer: 128 → num_classes
    
    Usage:
        head = StandardClassifierHead(in_features=1280, num_classes=8)
    """
    def __init__(self, in_features, num_classes=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.proj(x)
        return self.classifier(x)


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
    Enhanced QuanvTinyViT with flexible head options.
    
    Improvements:
    - Supports multiple head types: 'improved' (6-qubit VQC), 'advanced' (8-qubit VQC), 'standard' (classical)
    - 3 VQC layers (for quantum heads) or 2 hidden layers (for classical head)
    - Better projection and classification layers
    - Batch normalization for stability
    
    Usage:
        model = QuanvTinyViTImproved(num_classes=8, improvement_level='standard')  # Classical head
        model = QuanvTinyViTImproved(num_classes=8, improvement_level='improved')  # Quantum head (default)
    """
    def __init__(self, num_classes=8, pretrained=True, vqc_layers=3, improvement_level='standard'):
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

        # Select head type
        if improvement_level == 'standard':
            # Classical neural network classifier (no quantum circuit)
            self.head = StandardClassifierHead(
                in_features=self.base.num_features,
                num_classes=num_classes
            )
        elif improvement_level == 'advanced':
            # Advanced quantum head (8 qubits)
            self.head = VQCHeadAdvanced(
                in_features=self.base.num_features,
                num_classes=num_classes,
                n_qubits=8,
                n_layers=vqc_layers
            )
        else:  # improved (default)
            # Improved quantum head (6 qubits)
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