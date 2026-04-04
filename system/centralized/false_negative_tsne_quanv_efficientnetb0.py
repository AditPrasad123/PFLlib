import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from quanv_efficientnet_b0 import QuanvEfficientNetB0Improved
from utils.data_utils import read_client_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="False-negative analysis and t-SNE for centralized QuanvEfficientNetB0."
    )
    parser.add_argument("--dataset", type=str, default="ISIC2019_quanv")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(SCRIPT_DIR, "quanv_efficientnetb0_outputs", "best_quanv_efficientnet_b0_isic2019.pth"),
    )
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tsne_samples", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "quanv_efficientnetb0_outputs", "analysis_fn_tsne"),
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


def load_model(checkpoint_path, num_classes, device):
    model = QuanvEfficientNetB0Improved(
        num_classes=num_classes,
        pretrained=True,
        improvement_level="standard",
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_test_data(dataset_name):
    all_samples = []
    client_id = 0
    while True:
        try:
            ds = read_client_data(dataset_name, client_id, is_train=False)
            all_samples.extend(list(ds))
            client_id += 1
        except Exception:
            break

    if not all_samples:
        raise RuntimeError(f"No test data found for dataset '{dataset_name}'.")

    x_list = [s[0] for s in all_samples]
    y_list = [int(s[1]) for s in all_samples]
    return x_list, np.array(y_list, dtype=np.int64)


def batched_iter(x_list, y_np, batch_size):
    n = len(y_np)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.stack([x_list[i] for i in range(start, end)], dim=0)
        y = torch.tensor(y_np[start:end], dtype=torch.long)
        yield x, y, start, end


def collect_predictions_and_embeddings(model, x_list, y_true, batch_size, device):
    head = getattr(model, "head", None)
    if head is None:
        raise RuntimeError("Model has no 'head' module for embedding capture.")

    captured = {"emb": None}

    def _capture_head_input(module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["emb"] = inputs[0].detach()

    hook_handle = head.register_forward_pre_hook(_capture_head_input)

    records = []
    all_embeddings = []
    all_preds = []

    with torch.no_grad():
        for x, y, start, end in batched_iter(x_list, y_true, batch_size):
            x, y = move_batch_to_device(x, y, device)
            captured["emb"] = None

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if captured["emb"] is None:
                raise RuntimeError("Failed to capture embeddings from head input.")

            emb = captured["emb"].cpu().numpy()
            p_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()
            pr_np = probs.cpu().numpy()

            for i in range(len(y_np)):
                true_label = int(y_np[i])
                pred_label = int(p_np[i])
                confidence = float(np.max(pr_np[i]))
                is_fn = int(true_label != pred_label)
                records.append(
                    {
                        "sample_index": start + i,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "is_false_negative": is_fn,
                    }
                )

            all_embeddings.append(emb)
            all_preds.append(p_np)

    hook_handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    return records, embeddings, preds


def save_prediction_csvs(records, output_dir):
    all_csv = os.path.join(output_dir, "all_predictions_centralized.csv")
    fn_csv = os.path.join(output_dir, "false_negatives_centralized.csv")

    fields = ["sample_index", "true_label", "pred_label", "confidence", "is_false_negative"]
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    with open(fn_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            if r["is_false_negative"] == 1:
                writer.writerow(r)


def build_fn_summary(y_true, y_pred):
    total_by_true = Counter()
    fn_by_true = Counter()
    confusion = defaultdict(Counter)

    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)
        total_by_true[t] += 1
        if t != p:
            fn_by_true[t] += 1
            confusion[t][p] += 1

    rows = []
    for cls in sorted(total_by_true.keys()):
        total = total_by_true[cls]
        fn = fn_by_true[cls]
        fn_rate = (fn / total) if total > 0 else 0.0
        top_confused = confusion[cls].most_common(3)
        rows.append(
            {
                "true_label": cls,
                "total_samples": total,
                "false_negatives": fn,
                "fn_rate": fn_rate,
                "top_confused_predictions": "; ".join([f"{k}:{v}" for k, v in top_confused]),
            }
        )
    return rows


def save_fn_summary(summary_rows, output_dir):
    path = os.path.join(output_dir, "false_negative_summary_centralized.csv")
    fields = ["true_label", "total_samples", "false_negatives", "fn_rate", "top_confused_predictions"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def compute_tsne(embeddings, labels, preds, max_samples, perplexity, seed):
    n = embeddings.shape[0]
    if n < 3:
        raise RuntimeError("Need at least 3 samples to run t-SNE.")

    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        preds = preds[idx]

    pca_dim = min(50, embeddings.shape[1], max(2, embeddings.shape[0] - 1))
    emb_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(embeddings)

    safe_perplexity = max(2.0, min(perplexity, float(embeddings.shape[0] - 1)))
    emb_2d = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(emb_pca)

    return emb_2d, labels, preds


def save_tsne_plots(emb_2d, labels, preds, output_dir):
    incorrect = labels != preds

    plt.figure(figsize=(11, 9))
    sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, s=18, alpha=0.75, cmap="tab10")
    plt.colorbar(sc, label="True class")
    plt.title("Centralized t-SNE (color = true class)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_by_true_class_centralized.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(11, 9))
    plt.scatter(emb_2d[~incorrect, 0], emb_2d[~incorrect, 1], c="lightgray", s=16, alpha=0.35, label="Correct")
    plt.scatter(emb_2d[incorrect, 0], emb_2d[incorrect, 1], c="red", s=22, alpha=0.85, label="False negatives")
    plt.legend(loc="best")
    plt.title("Centralized t-SNE with false negatives highlighted")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_false_negatives_centralized.png"), dpi=220)
    plt.close()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    checkpoint = os.path.abspath(args.checkpoint)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading centralized checkpoint: {checkpoint}")
    model = load_model(checkpoint, args.num_classes, device)

    x_list, y_true = collect_test_data(args.dataset)
    records, embeddings, preds = collect_predictions_and_embeddings(
        model=model,
        x_list=x_list,
        y_true=y_true,
        batch_size=args.batch_size,
        device=device,
    )

    save_prediction_csvs(records, output_dir)
    save_fn_summary(build_fn_summary(y_true, preds), output_dir)

    emb_2d, tsne_labels, tsne_preds = compute_tsne(
        embeddings=embeddings,
        labels=y_true,
        preds=preds,
        max_samples=args.max_tsne_samples,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    save_tsne_plots(emb_2d, tsne_labels, tsne_preds, output_dir)

    fn_count = int(np.sum(y_true != preds))
    total = len(y_true)
    print("\nCentralized analysis complete.")
    print(f"Total test samples: {total}")
    print(f"False negatives: {fn_count}")
    print(f"False negative rate: {fn_count / max(1, total):.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
