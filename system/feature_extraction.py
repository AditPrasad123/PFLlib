import torch
from tqdm import tqdm

def extract_features(model, dataloader, device):
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting features"):
            x = x.to(device)

            backbone = model.base  # SkinLesionModel

            # Step 1: CNN + pooling → (B, 1280, 1, 1)
            feat = backbone.base(x)

            # 🔥 Step 2: flatten → (B, 1280)
            feat = feat.view(feat.size(0), -1)

            # Step 3: projector → (B, 8)
            feat = backbone.projector(feat)

            features.append(feat.cpu())
            labels.append(y)

    X = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    return X, y