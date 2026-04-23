import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from utils.data_utils import read_client_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="True class-specific Grad-CAM for centralized EfficientNetB0 on ISIC2019."
    )
    parser.add_argument("--dataset", type=str, default="ISIC2019")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(SCRIPT_DIR, "efficientnetb0_outputs", "best_efficientnet_b0_isic2019.pth"),
    )
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples_per_class", type=int, default=3)
    parser.add_argument("--only_false_negatives", action="store_true")
    parser.add_argument("--only_correct", action="store_true")
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overlay_on_original",
        action="store_true",
        help="Overlay Grad-CAM on the denormalized model input image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "efficientnetb0_outputs", "gradcam_outputs"),
    )
    return parser.parse_args()


def move_batch_to_device(x, y, device):
    if isinstance(x, list):
        x[0] = x[0].to(device)
        x = x[0]
    else:
        x = x.to(device)
    y = y.to(device)
    return x, y


def infer_num_classes(state_dict):
    for key in ("classifier.1.weight", "classifier.1.bias", "classifier.weight", "classifier.bias"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            if len(state_dict[key].shape) >= 1:
                return int(state_dict[key].shape[0])
    raise RuntimeError("Could not infer the number of classes from the checkpoint state dict.")


def load_model(checkpoint_path, device, num_classes=-1):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if num_classes is None or int(num_classes) <= 0:
        num_classes = infer_num_classes(state)

    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, num_classes


def find_target_conv_layer(model):
    conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise RuntimeError("No Conv2d layer found in EfficientNetB0 backbone.")
    return conv_layers[-1]


def denormalize_imagenet(x3):
    x = x3.detach().cpu().float().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = np.transpose(x, (1, 2, 0))
    x = (x * std) + mean
    return np.clip(x, 0.0, 1.0)


def apply_colormap(cam_2d):
    cmap = plt.get_cmap("jet")
    rgba = cmap(cam_2d)
    return rgba[..., :3].astype(np.float32)


def overlay_heatmap(image_rgb, cam_2d, alpha=0.45):
    heat = apply_colormap(cam_2d)
    return np.clip((1.0 - alpha) * image_rgb + alpha * heat, 0.0, 1.0)


def collect_predictions(model, dataset, batch_size, device):
    records = []

    client_id = 0
    while True:
        try:
            ds = read_client_data(dataset, client_id, is_train=False)
        except Exception:
            break

        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

        with torch.no_grad():
            sample_idx = 0
            for x, y in loader:
                x, y = move_batch_to_device(x, y, device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                x_cpu = x.detach().cpu()
                y_np = y.detach().cpu().numpy()
                p_np = preds.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()

                for i in range(len(y_np)):
                    t = int(y_np[i])
                    p = int(p_np[i])
                    records.append(
                        {
                            "client_id": client_id,
                            "sample_index": sample_idx,
                            "x": x_cpu[i],
                            "true_label": t,
                            "pred_label": p,
                            "confidence": float(np.max(probs_np[i])),
                            "is_false_negative": int(t != p),
                        }
                    )
                    sample_idx += 1

        client_id += 1

    if not records:
        raise RuntimeError("No records collected from test dataset.")

    return records


def select_samples(records, samples_per_class, only_false_negatives, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)

    for r in records:
        if only_false_negatives and r["is_false_negative"] == 0:
            continue
        by_class[r["true_label"]].append(r)

    selected = []
    for cls in sorted(by_class.keys()):
        items = sorted(by_class[cls], key=lambda z: (z["is_false_negative"] == 0, z["confidence"]))
        if len(items) <= samples_per_class:
            selected.extend(items)
            continue

        top = items[: max(samples_per_class * 2, samples_per_class)]
        idx = rng.choice(len(top), size=samples_per_class, replace=False)
        selected.extend([top[i] for i in sorted(idx)])

    return selected


def select_samples_with_mode(records, samples_per_class, only_false_negatives, only_correct, min_confidence, seed):
    if only_false_negatives and only_correct:
        raise ValueError("--only_false_negatives and --only_correct cannot be used together.")

    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("--min_confidence must be between 0.0 and 1.0.")

    filtered = [r for r in records if float(r["confidence"]) >= float(min_confidence)]

    if only_correct:
        filtered = [r for r in filtered if int(r["true_label"]) == int(r["pred_label"])]
        return select_samples(
            records=filtered,
            samples_per_class=samples_per_class,
            only_false_negatives=False,
            seed=seed,
        )

    return select_samples(
        records=filtered,
        samples_per_class=samples_per_class,
        only_false_negatives=only_false_negatives,
        seed=seed,
    )


def gradcam_for_sample(model, target_layer, x_single, class_idx, device):
    activations = {}
    gradients = {}

    def fwd_hook(module, inputs, output):
        activations["value"] = output
        if isinstance(output, torch.Tensor):
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    h1 = target_layer.register_forward_hook(fwd_hook)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        x = x_single.unsqueeze(0).to(device)
        logits = model(x)
        score = logits[0, class_idx]
        score.backward()

    h1.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

    acts = activations["value"]
    grads = gradients["value"]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

    cam = cam[0, 0].detach().cpu().numpy()
    if np.max(cam) > np.min(cam):
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    else:
        cam = np.zeros_like(cam)

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    return cam, pred, conf


def save_outputs(model, target_layer, selected, output_dir, device, overlay_on_original=False):
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    manifest = []
    for i, item in enumerate(selected):
        x = item["x"]
        t = int(item["true_label"])

        cam, p, conf = gradcam_for_sample(
            model=model,
            target_layer=target_layer,
            x_single=x,
            class_idx=t,
            device=device,
        )

        if overlay_on_original:
            rgb = denormalize_imagenet(x)
            input_title = "Input (denormalized RGB)"
            domain = "original_rgb"
        else:
            rgb = denormalize_imagenet(x)
            input_title = "Input (denormalized RGB)"
            domain = "original_rgb"

        ov = overlay_heatmap(rgb, cam)
        fn_flag = int(t != p)

        fname = f"{i:04d}_client{item['client_id']}_idx{item['sample_index']}_t{t}_p{p}_fn{fn_flag}.png"
        fpath = os.path.join(image_dir, fname)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title(input_title, fontsize=14, pad=12)
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM", fontsize=14, pad=12)
        axes[1].axis("off")

        axes[2].imshow(ov)
        axes[2].set_title(f"Overlay | true={t} pred={p} conf={conf:.3f}", fontsize=13, pad=12)
        axes[2].axis("off")

        plt.savefig(fpath, dpi=220, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

        manifest.append(
            {
                "file": os.path.basename(fpath),
                "client_id": item["client_id"],
                "sample_index": item["sample_index"],
                "true_label": t,
                "pred_label": p,
                "confidence": conf,
                "is_false_negative": fn_flag,
                "visualization_domain": domain,
            }
        )

    csv_path = os.path.join(output_dir, "gradcam_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "client_id",
                "sample_index",
                "true_label",
                "pred_label",
                "confidence",
                "is_false_negative",
                "visualization_domain",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    checkpoint = os.path.abspath(args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading centralized checkpoint: {checkpoint}")
    model, inferred_num_classes = load_model(checkpoint, device, num_classes=args.num_classes)
    print(f"Model num_classes: {inferred_num_classes}")

    for p in model.parameters():
        p.requires_grad = True

    target_layer = find_target_conv_layer(model)
    print(f"Grad-CAM target layer: {target_layer.__class__.__name__}")

    records = collect_predictions(
        model=model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        device=device,
    )

    selected = select_samples_with_mode(
        records=records,
        samples_per_class=args.samples_per_class,
        only_false_negatives=args.only_false_negatives,
        only_correct=args.only_correct,
        min_confidence=args.min_confidence,
        seed=args.seed,
    )

    if not selected:
        raise RuntimeError("No samples selected. Try removing --only_false_negatives.")

    save_outputs(
        model=model,
        target_layer=target_layer,
        selected=selected,
        output_dir=output_dir,
        device=device,
        overlay_on_original=args.overlay_on_original,
    )

    total = len(records)
    fn_count = sum(r["is_false_negative"] for r in records)
    print("\nCentralized EfficientNetB0 Grad-CAM analysis complete.")
    print(f"Total evaluated samples: {total}")
    print(f"False negatives in evaluated set: {fn_count}")
    print(f"Saved Grad-CAM samples: {len(selected)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()