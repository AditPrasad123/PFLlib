"""
Hybrid CNN-Transformer Backbone for Federated Learning

Architecture: EfficientNet-B4 (CNN) → Token Projection → Transformer Encoder
- BASE: CNN + Token Projection (shared, sent to server for aggregation)
- HEAD: Linear classifier (local, personalized per client)

This design is optimized for FedBABU where the backbone extracts features
that are shared globally while the head is personalized locally.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4


'''
base = cnn + patch + vit
head = classifier
'''

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_b4(weights="IMAGENET1K_V1")
        self.features = model.features[:5]  # cut early

    def forward(self, x):
        return self.features(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_ch, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.proj(x)
        return x

class TinyViTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class HybridBase(nn.Module):
    def __init__(self, cnn, patch, vit):
        super().__init__()
        self.cnn = cnn
        self.patch = patch
        self.vit = vit

    def forward(self, x):
        x = self.cnn(x)
        x = self.patch(x)
        x = self.vit(x)
        x = x.mean(dim=1)   # (B, 192)
        return x

class HybridModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # ============ CNN Feature Extractor ============
        # Pre-trained EfficientNet-B4 extracts spatial features
        # Uses ImageNet weights for strong initialization on visual tasks
        backbone = efficientnet_b4(weights="IMAGENET1K_V1")
        self.cnn = backbone.features          # (B, 1792, 7, 7)
        self.token_proj = nn.Conv2d(1792, 256, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,                    # Embedding dimension
            nhead=8,                        # Number of attention heads
            dim_feedforward=1024,           # Hidden dimension in FFN (4x expansion)
            batch_first=True                # Input shape: (Batch, Seq, Features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # ============ Classifier Head (FedBABU Local) ============
        # Personalized classifier trained locally, not shared with server
        # Enables local adaptation to client-specific label distributions
        self.head = nn.Linear(256, num_classes)

        # ============ Base Module (FedBABU Shared) ============
        # Groups CNN + Transformer as the "BODY" for server aggregation
        # This is the component that will be averaged across clients in federated learning
        self.base = nn.Sequential(
            self.cnn,
            self.token_proj
        )

    def forward(self, x):
        """
        Forward pass returns feature embeddings (NOT logits).
        
        The head is applied separately to enable flexible training strategies:
        - Joint training (body + head updates aggregated/local)
        - Head-only fine-tuning (body frozen, head personalized)
        
        Args:
            x: Input tensor (B, 3, 224, 224)
        
        Returns:
            Feature embeddings (B, 256)
        """
        # ============ Stage 1: CNN Feature Extraction ============
        x = self.cnn(x)  # (B, 1792, 7, 7)
        
        # ============ Stage 2: Token Projection ============
        x = self.token_proj(x)  # (B, 256, 7, 7)
        
        # ============ Stage 3: Reshape for Transformer ============
        # Convert spatial feature map to sequence of tokens
        # (B, 256, 7, 7) → (B, 49, 256) where 49 = 7*7 patches
        x = x.flatten(2).transpose(1, 2)
        
        # ============ Stage 4: Transformer Encoding ============
        x = self.transformer(x)  # (B, 49, 256)
        
        # ============ Stage 5: Global Pooling ============
        # Aggregate all spatial token representations via mean pooling
        # Produces fixed-size feature vector independent of spatial resolution
        x = x.mean(dim=1)  # (B, 256)
        
        return x  # Feature embeddings (NOT logits - head applied separately)


# ============ DEPRECATED / EXPERIMENTAL CLASSES ============
# The following classes were exploratory implementations and are not used
# in the current production model. Kept for reference/future development.

# class CNNBackbone(nn.Module):
#     """Early experimental CNN-only backbone (superseded by full hybrid model)"""
#     def __init__(self):
#         super().__init__()
#         model = efficientnet_b4(weights="IMAGENET1K_V1")
#         self.features = model.features[:5]
#
#     def forward(self, x):
#         return self.features(x)
#
# class PatchEmbed(nn.Module):
#     """Early experimental patch embedding (replaced by Conv2d token projection)"""
#     def __init__(self, in_ch, embed_dim):
#         super().__init__()
#         self.proj = nn.Linear(in_ch, embed_dim)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x
#
# class TinyViTBlock(nn.Module):
#     """Early experimental transformer block (replaced by nn.TransformerEncoder)"""
#     def __init__(self, dim, heads):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Linear(dim * 2, dim)
#         )
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#         x = x + self.mlp(self.norm2(x))
#         return x
#
# class Classifier(nn.Module):
#     """Early experimental classifier (now integrated as head in HybridModel)"""
#     def __init__(self, dim, num_classes):
#         super().__init__()
#         self.fc = nn.Linear(dim, num_classes)
#
#     def forward(self, x):
#         return self.fc(x)
#
# class HybridBase(nn.Module):
#     """Early experimental wrapper (replaced by direct sequential composition)"""
#     def __init__(self, cnn, patch, vit):
#         super().__init__()
#         self.cnn = cnn
#         self.patch = patch
#         self.vit = vit
#
#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.patch(x)
#         x = self.vit(x)
#         x = x.mean(dim=1)
#         return x


if __name__ == "__main__":
    """
    Verification script to validate model architecture and output shapes.
    
    Expected outputs:
    - features.shape: torch.Size([10, 256])  ← Feature embeddings
    - logits.shape: torch.Size([10, 8])      ← Classification logits
    """
    model = HybridModel(num_classes=8)
    x = torch.randn(10, 3, 224, 224)

    # ============ Forward Pass ============
    features = model(x)
    print(f"Feature embedding shape: {features.shape}")  # (10, 256)
    assert features.shape == (10, 256), "Feature shape mismatch!"

    # ============ Head Forward (Applied Separately) ============
    logits = model.head(features)
    print(f"Logits shape: {logits.shape}")  # (10, 8)
    assert logits.shape == (10, 8), "Logits shape mismatch!"
    
    print("\n✅ All architecture checks passed!")
        


