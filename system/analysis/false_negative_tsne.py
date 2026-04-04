import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(SCRIPT_DIR)
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from utils.data_utils import read_client_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="False-negative analysis and t-SNE visualization for federated model checkpoints."
    )
    parser.add_argument("--dataset", type=str, default="ISIC2019_quanv")
    parser.add_argument("--algorithm", type=str, default="FedProx")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tsne_samples", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def get_default_paths(dataset, algorithm):
    model_path = os.path.join(SYSTEM_DIR, "models", dataset, f"{algorithm}_server.pt")
    output_dir = os.path.join(SYSTEM_DIR, "..", "results", f"analysis_{dataset}_{algorithm}")
    return model_path, output_dir


def load_num_clients(dataset):
    cfg_path = os.path.join(SYSTEM_DIR, "..", "dataset", dataset, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["num_clients"])


def get_head_module(model):
    for name in ("head", "fc", "classifier"):
        if hasattr(model, name):
            return getattr(model, name)
    return None


def move_batch_to_device(x, y, device):
    if isinstance(x, list):
        x[0] = x[0].to(device)
        x = x[0]
    else:
        x = x.to(device)
    y = y.to(device)
    return x, y


def collect_predictions_and_embeddings(model, dataset, num_clients, batch_size, device):
    model.eval()

    head = get_head_module(model)
    if head is None:
        raise RuntimeError("Could not find model head (head/fc/classifier).")

    captured = {"emb": None}

    def _capture_head_input(module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["emb"] = inputs[0].detach()

    hook_handle = head.register_forward_pre_hook(_capture_head_input)

    records = []
    all_embeddings = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for client_id in range(num_clients):
            ds = read_client_data(dataset, client_id, is_train=False)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

            sample_index = 0
            for x, y in loader:
                x, y = move_batch_to_device(x, y, device)
                captured["emb"] = None
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                if captured["emb"] is None:
                    raise RuntimeError("Failed to capture embeddings from model head input.")

                emb = captured["emb"].detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                p_np = preds.detach().cpu().numpy()
                pr_np = probs.detach().cpu().numpy()

                for i in range(len(y_np)):
                    true_label = int(y_np[i])
                    pred_label = int(p_np[i])
                    confidence = float(np.max(pr_np[i]))
                    is_fn = int(true_label != pred_label)

                    records.append(
                        {
                            "client_id": client_id,
                            "sample_index": sample_index,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "confidence": confidence,
                            "is_false_negative": is_fn,
                        }
                    )
                    sample_index += 1

                all_embeddings.append(emb)
                all_labels.append(y_np)
                all_preds.append(p_np)

    hook_handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    return records, embeddings, labels, preds


def save_prediction_csvs(records, output_dir):
    all_csv = os.path.join(output_dir, "all_predictions.csv")
    fn_csv = os.path.join(output_dir, "false_negatives.csv")

    fields = ["client_id", "sample_index", "true_label", "pred_label", "confidence", "is_false_negative"]

    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with open(fn_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            if r["is_false_negative"] == 1:
                writer.writerow(r)


def build_fn_summary(records):
    total_by_true = Counter()
    fn_by_true = Counter()
    confusion = defaultdict(Counter)

    for r in records:
        t = r["true_label"]
        p = r["pred_label"]
        total_by_true[t] += 1
        if t != p:
            fn_by_true[t] += 1
            confusion[t][p] += 1

    rows = []
    classes = sorted(total_by_true.keys())
    for cls in classes:
        total = total_by_true[cls]
        fn = fn_by_true[cls]
        fn_rate = (fn / total) if total > 0 else 0.0
        top_confused = confusion[cls].most_common(3)
        top_confused_str = "; ".join([f"{k}:{v}" for k, v in top_confused]) if top_confused else ""
        rows.append(
            {
                "true_label": cls,
                "total_samples": total,
                "false_negatives": fn,
                "fn_rate": fn_rate,
                "top_confused_predictions": top_confused_str,
            }
        )
    return rows


def save_fn_summary(summary_rows, output_dir):
    path = os.path.join(output_dir, "false_negative_summary.csv")
    fields = ["true_label", "total_samples", "false_negatives", "fn_rate", "top_confused_predictions"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)


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
    plt.title("t-SNE of test embeddings (color = true class)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_by_true_class.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(11, 9))
    plt.scatter(emb_2d[~incorrect, 0], emb_2d[~incorrect, 1], c="lightgray", s=16, alpha=0.35, label="Correct")
    plt.scatter(emb_2d[incorrect, 0], emb_2d[incorrect, 1], c="red", s=22, alpha=0.85, label="False negatives")
    plt.legend(loc="best")
    plt.title("t-SNE with false negatives highlighted")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_false_negatives.png"), dpi=220)
    plt.close()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    default_model_path, default_output_dir = get_default_paths(args.dataset, args.algorithm)
    model_path = args.model_path if args.model_path else default_model_path
    output_dir = args.output_dir if args.output_dir else default_output_dir
    model_path = os.path.abspath(model_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)

    num_clients = load_num_clients(args.dataset)
    print(f"Running analysis for dataset={args.dataset}, algorithm={args.algorithm}, clients={num_clients}")

    records, embeddings, labels, preds = collect_predictions_and_embeddings(
        model=model,
        dataset=args.dataset,
        num_clients=num_clients,
        batch_size=args.batch_size,
        device=device,
    )

    save_prediction_csvs(records, output_dir)
    fn_summary = build_fn_summary(records)
    save_fn_summary(fn_summary, output_dir)

    emb_2d, tsne_labels, tsne_preds = compute_tsne(
        embeddings=embeddings,
        labels=labels,
        preds=preds,
        max_samples=args.max_tsne_samples,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    save_tsne_plots(emb_2d, tsne_labels, tsne_preds, output_dir)

    total = len(records)
    fn_count = int(np.sum(labels != preds))
    print("\nAnalysis complete.")
    print(f"Total test samples: {total}")
    print(f"False negatives: {fn_count}")
    print(f"False negative rate: {fn_count / max(1, total):.4f}")
    print(f"Outputs saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
