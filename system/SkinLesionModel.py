import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=9):
        super(SkinLesionModel, self).__init__()

        # Load pretrained EfficientNetB0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)

        self.base = nn.Sequential(
                  backbone.features,
                  nn.AdaptiveAvgPool2d(1)
                  )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 8)
        )

        # Classifier
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.base(x)
        x = torch.flatten(x, 1)

        # Projection
        x = self.projector(x)

        # Classification
        logits = self.fc(x)

        return logits