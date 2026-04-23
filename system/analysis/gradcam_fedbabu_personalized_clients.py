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
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(SCRIPT_DIR)
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from utils.data_utils import read_client_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="True class-specific Grad-CAM for FedBABU personalized client models."
    )
    parser.add_argument("--dataset", type=str, default="ISIC2019_quanv")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--clients_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples_per_class", type=int, default=3)
    parser.add_argument("--only_false_negatives", action="store_true")
    parser.add_argument("--only_correct", action="store_true")
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--overlay_on_original",
        action="store_true",
        help="Map quanv-space CAM to original ISIC image and overlay there.",
    )
    parser.add_argument(
        "--original_dataset_dir",
        type=str,
        default="",
        help="Path to dataset root that contains ISIC2019/train and ISIC2019/test.",
    )
    return parser.parse_args()


def default_paths(dataset, run_id):
    clients_dir = os.path.join(
        SYSTEM_DIR,
        "models",
        dataset,
        "FedBABU_personalized_clients",
        f"run_{run_id}",
    )
    output_dir = os.path.join(
        SYSTEM_DIR,
        "..",
        "results",
        f"gradcam_{dataset}_FedBABU_personalized_run_{run_id}",
    )
    return os.path.abspath(clients_dir), os.path.abspath(output_dir)


def find_target_conv_layer(model):
    if not hasattr(model, "base"):
        raise RuntimeError("Could not locate a base backbone on the loaded model.")

    feature_backbone = model.base

    # Quanv models store the backbone inside a Sequential wrapper.
    if isinstance(feature_backbone, nn.Sequential) and len(feature_backbone) > 0:
        feature_backbone = feature_backbone[0]
    # Classical FedBABU checkpoints wrap a torchvision EfficientNetB0 inside BaseHeadSplit.
    # That backbone exposes .features directly instead of being indexable.
    elif hasattr(feature_backbone, "features"):
        feature_backbone = feature_backbone.features

    conv_layers = [m for m in feature_backbone.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise RuntimeError("No Conv2d layer found in feature backbone.")
    return conv_layers[-1]


def move_batch_to_device(x, y, device):
    if isinstance(x, list):
        x[0] = x[0].to(device)
        x = x[0]
    else:
        x = x.to(device)
    y = y.to(device)
    return x, y


def tensor_to_pseudorgb(x4):
    x = x4.detach().cpu().float().numpy()
    if x.shape[0] >= 3:
        rgb = x[:3]
    else:
        rgb = np.repeat(x[:1], 3, axis=0)

    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        mn = float(np.min(rgb[c]))
        mx = float(np.max(rgb[c]))
        if mx > mn:
            out[c] = (rgb[c] - mn) / (mx - mn)
        else:
            out[c] = 0.0
    return np.transpose(out, (1, 2, 0))


def apply_colormap(cam_2d):
    cmap = plt.get_cmap("jet")
    rgba = cmap(cam_2d)
    return rgba[..., :3].astype(np.float32)


def overlay_heatmap(image_rgb, cam_2d, alpha=0.45):
    heat = apply_colormap(cam_2d)
    return np.clip((1.0 - alpha) * image_rgb + alpha * heat, 0.0, 1.0)


def normalize_rgb_image(img):
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3].astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def map_cam_to_original_image(cam_2d, original_hw):
    # CAM is produced in quanv input space (24x24). Map to preprocessing size (48x48)
    # and then to original image resolution for visualization.
    cam_48 = cv2.resize(cam_2d, (48, 48), interpolation=cv2.INTER_LINEAR)
    h, w = original_hw
    cam_orig = cv2.resize(cam_48, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.max(cam_orig) > np.min(cam_orig):
        cam_orig = (cam_orig - np.min(cam_orig)) / (np.max(cam_orig) - np.min(cam_orig))
    else:
        cam_orig = np.zeros_like(cam_orig)
    return cam_orig


def load_original_rgb(client_id, sample_index, dataset_root, cache, split="test"):
    key = (client_id, split)
    if key not in cache:
        npz_path = os.path.join(dataset_root, "ISIC2019", split, f"{client_id}.npz")
        if not os.path.exists(npz_path):
            cache[key] = None
        else:
            data = np.load(npz_path, allow_pickle=True)
            cache[key] = data["data"].item().get("x", None)

    images = cache.get(key, None)
    if images is None or sample_index < 0 or sample_index >= len(images):
        return None

    return normalize_rgb_image(images[sample_index])


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


def collect_predictions_for_client(model, client_id, dataset, batch_size, device):
    model.eval()
    ds = read_client_data(dataset, client_id, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    records = []
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


def select_samples_with_mode(
    records,
    samples_per_class,
    only_false_negatives,
    only_correct,
    min_confidence,
    seed,
):
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


def save_outputs_for_client(
    model,
    target_layer,
    selected,
    out_dir,
    device,
    overlay_on_original=False,
    original_dataset_dir=None,
    original_cache=None,
):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

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

        rgb = tensor_to_pseudorgb(x)
        cam_vis = cam
        input_title = "Input (pseudo-RGB)"
        cam_title = "Grad-CAM"
        domain = "quanv_pseudo_rgb"

        if overlay_on_original and original_dataset_dir and original_cache is not None:
            rgb_original = load_original_rgb(
                client_id=item["client_id"],
                sample_index=item["sample_index"],
                dataset_root=original_dataset_dir,
                cache=original_cache,
                split="test",
            )
            if rgb_original is not None:
                rgb = rgb_original
                cam_vis = map_cam_to_original_image(cam, rgb.shape[:2])
                input_title = "Input (original RGB)"
                cam_title = "Grad-CAM (mapped)"
                domain = "original_rgb_mapped"

        ov = overlay_heatmap(rgb, cam_vis)
        fn_flag = int(t != p)

        fname = f"{i:04d}_idx{item['sample_index']}_t{t}_p{p}_fn{fn_flag}.png"
        fpath = os.path.join(img_dir, fname)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title(input_title, fontsize=14, pad=12)
        axes[0].axis("off")

        axes[1].imshow(cam_vis, cmap="jet")
        axes[1].set_title(cam_title, fontsize=14, pad=12)
        axes[1].axis("off")

        axes[2].imshow(ov)
        axes[2].set_title(f"Overlay | true={t} pred={p} conf={conf:.3f}", fontsize=13, pad=12)
        axes[2].axis("off")

        plt.savefig(fpath, dpi=220, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

        manifest.append(
            {
                "file": os.path.basename(fpath),
                "sample_index": item["sample_index"],
                "true_label": t,
                "pred_label": p,
                "confidence": conf,
                "is_false_negative": fn_flag,
                "visualization_domain": domain,
            }
        )

    csv_path = os.path.join(out_dir, "gradcam_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
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

    default_clients_dir, default_output_dir = default_paths(args.dataset, args.run_id)
    clients_dir = os.path.abspath(args.clients_dir) if args.clients_dir else default_clients_dir
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    original_dataset_dir = (
        os.path.abspath(args.original_dataset_dir)
        if args.original_dataset_dir
        else os.path.abspath(os.path.join(SYSTEM_DIR, "..", "dataset"))
    )
    original_cache = {}

    pattern = os.path.join(clients_dir, "client_*.pt")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No personalized client checkpoints found in: {clients_dir}\n"
            "Run FedBABU once after enabling personalized checkpoint saving."
        )

    total_records = 0
    total_fn = 0
    total_saved = 0

    for ckpt in ckpts:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        client_id = int(name.split("_")[-1])

        print(f"Processing client {client_id}: {ckpt}")
        model = torch.load(ckpt, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        # Personalized models are often saved with frozen backbone params after TTFT.
        # Grad-CAM needs activation gradients at the target conv layer, so enable grads.
        for p in model.parameters():
            p.requires_grad = True

        target_layer = find_target_conv_layer(model)
        records = collect_predictions_for_client(
            model=model,
            client_id=client_id,
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

        client_out = os.path.join(output_dir, f"client_{client_id}")
        if selected:
            save_outputs_for_client(
                model=model,
                target_layer=target_layer,
                selected=selected,
                out_dir=client_out,
                device=device,
                overlay_on_original=args.overlay_on_original,
                original_dataset_dir=original_dataset_dir,
                original_cache=original_cache,
            )

        total_records += len(records)
        total_fn += sum(r["is_false_negative"] for r in records)
        total_saved += len(selected)

    print("\nTrue Grad-CAM analysis complete (personalized client checkpoints).")
    print(f"Total evaluated samples: {total_records}")
    print(f"Total false negatives: {total_fn}")
    print(f"Saved Grad-CAM samples: {total_saved}")
    print(f"Outputs saved to: {output_dir}")
    if args.overlay_on_original:
        print(
            "Note: CAM was computed on quanv tensors and mapped back to original image coordinates "
            "using the preprocessing geometry (24x24 -> 48x48 -> original size)."
        )


if __name__ == "__main__":
    main()
