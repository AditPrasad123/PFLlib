"""
Enhanced QuanvEfficientNetB0 with multiple head options: VQC (quantum) or Standard (classical).
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
    Enhanced QuanvEfficientNetB0 with flexible head options.
    
    Improvements:
    - Supports multiple head types: 'improved' (6-qubit VQC), 'advanced' (8-qubit VQC), 'standard' (classical)
    - 3 VQC layers (for quantum heads) or 2 hidden layers (for classical head)
    - Better projection and classification layers
    - Batch normalization for stability
    
    Usage:
        model = QuanvEfficientNetB0Improved(num_classes=8, improvement_level='standard')  # Classical head
        model = QuanvEfficientNetB0Improved(num_classes=8, improvement_level='improved')  # Quantum head (default)
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

        # Select head type
        in_features = backbone.classifier[1].in_features
        if improvement_level == 'standard':
            # Classical neural network classifier (no quantum circuit)
            self.head = StandardClassifierHead(
                in_features=in_features,
                num_classes=num_classes
            )
        elif improvement_level == 'advanced':
            # Advanced quantum head (8 qubits)
            self.head = VQCHeadAdvanced(
                in_features=in_features,
                num_classes=num_classes,
                n_qubits=8,
                n_layers=vqc_layers
            )
        else:  # improved (default)
            # Improved quantum head (6 qubits)
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