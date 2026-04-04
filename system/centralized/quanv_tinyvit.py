"""
Centralized training script for ISIC2019_quanv using TinyViT
Combines all federated clients' data into one dataset for centralized learning comparison
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import time
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             f1_score, recall_score, precision_score,
                             cohen_kappa_score, matthews_corrcoef,
                             roc_curve, precision_recall_curve, auc,
                             average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Allow importing model definitions from ../quanv_tinyvit.py when this
# script is run directly from the centralized folder.
SYSTEM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

from quanv_tinyvit import QuanvTinyViTImproved


class ISIC2019_QuanvCentralizedDataset(Dataset):
    """Combined ISIC2019_quanv dataset from all federated clients"""

    def __init__(self, data_dir='ISIC2019_quanv', train=True, transform=None):
        self.transform = transform
        self.train = train

        self.images = []
        self.labels = []

        split = 'train' if train else 'test'

        for client_id in range(6):
            npz_path = os.path.join(data_dir, split, f'{client_id}.npz')

            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} not found")
                continue

            data = np.load(npz_path, allow_pickle=True)
            client_data = data['data'].item()

            self.images.extend(client_data['x'])
            self.labels.extend(client_data['y'])

        self.labels = np.array(self.labels)

        print(f"Loaded {split} data: {len(self.images)} images, {len(np.unique(self.labels))} classes")
        print(f"Class distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Normalize layout to HWC with 4 channels before augmentation.
        if isinstance(image, np.ndarray) and image.ndim == 4 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        if isinstance(image, np.ndarray) and image.ndim == 3:
            if image.shape[0] == 4:   # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[-1] == 4:  # already HWC
                pass
            else:
                raise ValueError(f"Unexpected numpy image shape: {image.shape}")
        elif isinstance(image, np.ndarray) and image.ndim == 4:
            if image.shape[1] == 4:   # (1, C, H, W) -> HWC
                image = np.transpose(image[0], (1, 2, 0))
            elif image.shape[-1] == 4:  # (1, H, W, C) -> HWC
                image = image[0]
            else:
                raise ValueError(f"Unexpected numpy image shape: {image.shape}")
        else:
            raise ValueError(
                f"Unexpected image type/shape: type={type(image)}, shape={getattr(image, 'shape', None)}"
            )

        if image.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels, got shape {image.shape}")

        if self.transform:
            image = self.transform(image=image)['image']

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        # HWC -> CHW for model input (expects 4xHxW).
        image = np.transpose(image, (2, 0, 1))

        if image.shape[0] != 4:
            raise ValueError(f"Expected CHW with 4 channels, got {image.shape}")

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_transforms():
    """Define training/testing augmentations for 4-channel quanv tensors."""
    import albumentations

    # Keep crop size consistent with federated quanv pipeline input.
    # QuanvTinyViTImproved then upsamples internally to 224x224.
    sz = 24
    train_transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.3),
        albumentations.Rotate(limit=45, p=0.3),
        albumentations.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0),
        albumentations.RandomCrop(sz, sz),
    ])

    test_transform = albumentations.Compose([
        albumentations.PadIfNeeded(min_height=sz, min_width=sz, border_mode=0),
        albumentations.CenterCrop(sz, sz),
    ])

    return train_transform, test_transform, None


def stratified_split_indices(labels, train_ratio=0.8, seed=42):
    """Create stratified train/val split indices."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        if len(class_indices) == 1:
            train_indices.extend(class_indices.tolist())
            continue

        n_train = int(len(class_indices) * train_ratio)
        n_train = max(1, min(n_train, len(class_indices) - 1))

        train_indices.extend(class_indices[:n_train].tolist())
        val_indices.extend(class_indices[n_train:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def report_class_coverage(labels, num_classes, split_name):
    """Report class coverage and whether AUC is expected to be well-defined."""
    labels = np.asarray(labels)
    present = sorted(np.unique(labels).tolist())
    missing = sorted(set(range(num_classes)) - set(present))
    counts = np.bincount(labels, minlength=num_classes)

    print(f"[{split_name}] class counts: {counts}")
    if missing:
        print(f"[{split_name}] Warning: missing classes {missing}; multiclass AUC may be undefined (NaN).")
    else:
        print(f"[{split_name}] Class coverage OK for multiclass AUC.")

    return missing


def remap_labels_inplace(dataset, label_map):
    """Remap dataset labels in-place to contiguous indices."""
    dataset.labels = np.array(
        [label_map[int(label)] for label in dataset.labels], dtype=np.int64
    )


def compute_sensitivity_specificity(y_true, y_pred, num_classes):
    """Compute sensitivity and specificity for multi-class classification."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    sensitivities = []
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    macro_sensitivity = np.mean(sensitivities)
    macro_specificity = np.mean(specificities)

    class_counts = np.bincount(y_true, minlength=num_classes)
    weights = class_counts / class_counts.sum()

    weighted_sensitivity = np.average(sensitivities, weights=weights)
    weighted_specificity = np.average(specificities, weights=weights)

    tp_total = np.trace(cm)
    fn_total = cm.sum() - np.trace(cm)
    fp_total = cm.sum() - np.trace(cm)
    tn_total = (num_classes * cm.sum()) - tp_total - fn_total - fp_total

    micro_sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0

    return {
        'sensitivity_macro': macro_sensitivity,
        'sensitivity_micro': micro_sensitivity,
        'sensitivity_weighted': weighted_sensitivity,
        'specificity_macro': macro_specificity,
        'specificity_micro': micro_specificity,
        'specificity_weighted': weighted_specificity,
    }


def compute_accuracy_averages(y_true, y_pred, num_classes):
    """Compute micro and macro accuracy for multiclass predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    total = cm.sum()
    micro_accuracy = np.trace(cm) / total if total > 0 else 0.0

    class_totals = cm.sum(axis=1)
    class_accuracies = np.divide(
        np.diag(cm),
        class_totals,
        out=np.zeros_like(class_totals, dtype=np.float64),
        where=class_totals > 0,
    )
    macro_accuracy = np.mean(class_accuracies) if len(class_accuracies) > 0 else 0.0

    return {
        'accuracy_micro': micro_accuracy,
        'accuracy_macro': macro_accuracy,
    }


def compute_brier_score(y_true, y_pred_probs, num_classes):
    """Compute multiclass Brier Score."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=range(num_classes)).astype(np.float64)
    y_p = np.asarray(y_pred_probs, dtype=np.float64)
    return float(np.mean(np.sum((y_p - y_bin) ** 2, axis=1)))


def compute_model_complexity(model, input_size=(1, 4, 24, 24)):
    """Return total/trainable param counts and FLOPs."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        'total_params': total,
        'trainable_params': trainable,
        'non_trainable_params': total - trainable,
        'flops': None,
        'flops_str': 'N/A',
    }
    try:
        flop_count = [0]
        hooks = []

        def conv_hook(module, inp, out):
            b = inp[0].shape[0]
            c_in = module.in_channels
            c_out = module.out_channels
            kH, kW = (module.kernel_size if isinstance(module.kernel_size, tuple)
                      else (module.kernel_size, module.kernel_size))
            oH, oW = out.shape[2], out.shape[3]
            macs = b * c_out * oH * oW * (c_in // module.groups) * kH * kW
            flop_count[0] += 2 * macs

        def linear_hook(module, inp, out):
            b = inp[0].numel() // module.in_features
            flop_count[0] += 2 * b * module.in_features * module.out_features

        model_cpu = model.cpu()
        for m in model_cpu.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))

        model_cpu.eval()
        with torch.no_grad():
            model_cpu(torch.randn(input_size))

        for h in hooks:
            h.remove()

        flops = flop_count[0]
        result['flops'] = flops
        if flops >= 1e9:
            result['flops_str'] = f"{flops / 1e9:.3f} GFLOPs"
        elif flops >= 1e6:
            result['flops_str'] = f"{flops / 1e6:.3f} MFLOPs"
        else:
            result['flops_str'] = f"{flops:,} FLOPs"
    except Exception as exc:
        result['flops_str'] = f'N/A ({exc})'
    return result


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={'fontsize': 12},
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curves(y_true, y_probs, num_classes, save_path='roc_curves.png'):
    """Plot and save per-class ROC curves (one-vs-rest)."""
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=range(num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(num_classes):
        positives = y_bin[:, i].sum()
        negatives = y_bin.shape[0] - positives
        if positives == 0 or negatives == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})', lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()


def plot_pr_curves(y_true, y_probs, num_classes, save_path='pr_curves.png'):
    """Plot and save per-class Precision-Recall curves."""
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=range(num_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(num_classes):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f'Class {i} (AP = {pr_auc:.3f})', lw=2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PR curves saved to {save_path}")
    plt.close()


def plot_overall_roc_curve(y_true, y_probs, num_classes, save_path='overall_roc_curve.png'):
    """Plot and save overall (micro-average) ROC curve."""
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=range(num_classes))
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    try:
        auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr',
                                  average='macro', labels=np.arange(num_classes))
    except ValueError:
        auc_macro = np.nan

    try:
        auc_weighted = roc_auc_score(y_true, y_probs, multi_class='ovr',
                                     average='weighted', labels=np.arange(num_classes))
    except ValueError:
        auc_weighted = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_micro, tpr_micro, lw=2, label=f'Micro-average ROC (AUC = {auc_micro:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Overall ROC Curve', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    text = f"Macro AUC: {auc_macro:.3f}\nWeighted AUC: {auc_weighted:.3f}"
    ax.text(0.60, 0.20, text, transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overall ROC curve saved to {save_path}")
    plt.close()


def plot_overall_pr_curve(y_true, y_probs, num_classes, save_path='overall_pr_curve.png'):
    """Plot and save overall (micro-average) PR curve."""
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=range(num_classes))
    precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), y_probs.ravel())
    ap_micro = auc(recall_micro, precision_micro)

    try:
        ap_macro = average_precision_score(y_bin, y_probs, average='macro')
    except ValueError:
        ap_macro = np.nan

    try:
        ap_weighted = average_precision_score(y_bin, y_probs, average='weighted')
    except ValueError:
        ap_weighted = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_micro, precision_micro, lw=2, label=f'Micro-average PR (AP = {ap_micro:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Overall Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    text = f"Macro AP: {ap_macro:.3f}\nWeighted AP: {ap_weighted:.3f}"
    ax.text(0.60, 0.20, text, transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overall PR curve saved to {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch,
                         save_path='training_curves.png'):
    """Plot train/val loss and accuracy side-by-side over epochs."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-o', markersize=4, linewidth=1.5, label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-o', markersize=4, linewidth=1.5, label='Val Loss')
    ax1.axvline(x=best_epoch + 1, color='green', linestyle='--', linewidth=1.5,
                label=f'Best Epoch ({best_epoch + 1})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-o', markersize=4, linewidth=1.5, label='Train Acc')
    ax2.plot(epochs, val_accs,   'r-o', markersize=4, linewidth=1.5, label='Val Acc')
    ax2.axvline(x=best_epoch + 1, color='green', linestyle='--', linewidth=1.5,
                label=f'Best Epoch ({best_epoch + 1})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.suptitle('QuanvTinyViT – Training Curves', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_convergence_rate(train_losses, val_accs, save_path='convergence_rate.png'):
    """Plot training loss and validation accuracy across epochs side-by-side."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-o', markersize=4, linewidth=1.5, label='Train Loss')
    ax1.set_xticks(epochs)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accs, color='orange', marker='o', markersize=4, linewidth=1.5,
             label='Validation Accuracy')
    ax2.set_xticks(epochs)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.suptitle('QuanvTinyViT – Convergence Curves', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence rate graph saved to {save_path}")
    plt.close()


class CentralizedTrainer:
    def __init__(self, model, device, num_classes=9, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / len(train_loader), correct / total

    def evaluate(self, test_loader):
        """Evaluate on a data split."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        all_probs  = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds  = np.concatenate(all_preds, axis=0)

        present_classes = np.unique(all_labels)
        if len(present_classes) < self.num_classes:
            missing = sorted(set(range(self.num_classes)) - set(present_classes.tolist()))
            print(f"Warning: Missing classes in evaluation split: {missing}")

        try:
            auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr',
                                      average='macro', labels=np.arange(self.num_classes))
            auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr',
                                         average='weighted', labels=np.arange(self.num_classes))
            auc_val = auc_weighted
        except ValueError as e:
            print(f"Warning: Could not compute multiclass AUC: {e}")
            auc_macro = auc_weighted = auc_val = np.nan

        f1_macro    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
        f1_micro    = f1_score(all_labels, all_preds, average='micro',    zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        recall_macro    = recall_score(all_labels, all_preds, average='macro',    zero_division=0)
        recall_micro    = recall_score(all_labels, all_preds, average='micro',    zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        precision_macro    = precision_score(all_labels, all_preds, average='macro',    zero_division=0)
        precision_micro    = precision_score(all_labels, all_preds, average='micro',    zero_division=0)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)

        kappa = cohen_kappa_score(all_labels, all_preds)
        mcc   = matthews_corrcoef(all_labels, all_preds)
        brier = compute_brier_score(all_labels, all_probs, self.num_classes)

        sens_spec  = compute_sensitivity_specificity(all_labels, all_preds, self.num_classes)
        acc_scores = compute_accuracy_averages(all_labels, all_preds, self.num_classes)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'accuracy_micro': acc_scores['accuracy_micro'],
            'accuracy_macro': acc_scores['accuracy_macro'],
            'auc': auc_val,
            'auc_macro': auc_macro,
            'auc_weighted': auc_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'recall_weighted': recall_weighted,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'precision_weighted': precision_weighted,
            'kappa': kappa,
            'mcc': mcc,
            'brier_score': brier,
            'sensitivity_macro': sens_spec['sensitivity_macro'],
            'sensitivity_micro': sens_spec['sensitivity_micro'],
            'sensitivity_weighted': sens_spec['sensitivity_weighted'],
            'specificity_macro': sens_spec['specificity_macro'],
            'specificity_micro': sens_spec['specificity_micro'],
            'specificity_weighted': sens_spec['specificity_weighted'],
            'y_true': all_labels,
            'y_pred': all_preds,
            'y_probs': all_probs,
        }


def main():
    start_time = time.time()

    # Configuration
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS   = 30
    BATCH_SIZE   = 16
    LEARNING_RATE = 0.001
    NUM_CLASSES  = 8
    TRAIN_RATIO  = 0.8

    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Train-Test Ratio: {TRAIN_RATIO:.1%}-{1-TRAIN_RATIO:.1%}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'quanv_tinyvit_outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Dataset ---
    print("\n=== Loading ISIC2019_quanv Dataset ===")
    train_transform, test_transform, _ = get_transforms()

    data_dir = '../../dataset/ISIC2019_quanv'

    full_dataset = ISIC2019_QuanvCentralizedDataset(
        data_dir=data_dir, train=True, transform=train_transform
    )

    train_indices, val_indices = stratified_split_indices(
        full_dataset.labels, train_ratio=TRAIN_RATIO, seed=42
    )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    test_dataset = ISIC2019_QuanvCentralizedDataset(
        data_dir=data_dir, train=False, transform=test_transform
    )

    # Remap labels to contiguous range if any class is missing
    present_classes = sorted(
        np.unique(np.concatenate([full_dataset.labels, test_dataset.labels])).tolist()
    )
    if len(present_classes) < NUM_CLASSES:
        print(f"Warning: Dataset has only {len(present_classes)} effective classes: {present_classes}")
        print("Info: Remapping labels to contiguous range [0..K-1].")
        label_map = {old: new for new, old in enumerate(present_classes)}
        remap_labels_inplace(full_dataset, label_map)
        remap_labels_inplace(test_dataset, label_map)
        NUM_CLASSES = len(present_classes)
        print(f"Effective NUM_CLASSES set to {NUM_CLASSES}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    print("\n=== AUC Sanity Check (Before Training) ===")
    report_class_coverage(full_dataset.labels[train_indices], NUM_CLASSES, "Train")
    report_class_coverage(full_dataset.labels[val_indices],   NUM_CLASSES, "Val")
    report_class_coverage(test_dataset.labels,                NUM_CLASSES, "Test")

    # --- Model ---
    print("\n=== Creating QuanvTinyViT Model ===")
    model = QuanvTinyViTImproved(
        num_classes=NUM_CLASSES,
        pretrained=True,
        improvement_level='standard'   # classical head — no QNN overhead
    )

    print("\n=== Model Complexity ===")
    complexity = compute_model_complexity(model, input_size=(1, 4, 24, 24))
    print(f"  Total parameters:         {complexity['total_params']:,} ({complexity['total_params']/1e6:.3f}M)")
    print(f"  Trainable parameters:     {complexity['trainable_params']:,} ({complexity['trainable_params']/1e6:.3f}M)")
    print(f"  Non-trainable parameters: {complexity['non_trainable_params']:,} ({complexity['non_trainable_params']/1e6:.3f}M)")
    print(f"  FLOPs (single inference): {complexity['flops_str']}")

    # --- Training ---
    trainer = CentralizedTrainer(model, DEVICE, NUM_CLASSES, LEARNING_RATE)

    print("\n=== Training ===")
    best_val_acc = 0
    best_epoch   = 0
    best_model_path = os.path.join(output_dir, 'best_quanv_tinyvit_isic2019.pth')

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_metrics = trainer.evaluate(val_loader)
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val AUC: {val_metrics['auc']:.4f}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch   = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"\u2713 Best model saved (Acc: {val_metrics['accuracy']:.4f})")

        trainer.scheduler.step()

    # --- Test ---
    print("\n=== Testing (Best Model) ===")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = trainer.evaluate(test_loader)

    print(f"Best Epoch: {best_epoch+1}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Accuracy (Micro): {test_metrics['accuracy_micro']:.4f}")
    print(f"Test Accuracy (Macro): {test_metrics['accuracy_macro']:.4f}")

    print("\n=== Detailed Test Metrics ===")
    print("Accuracy:")
    print(f"  ├─ Micro:    {test_metrics['accuracy_micro']:.4f}")
    print(f"  └─ Macro:    {test_metrics['accuracy_macro']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"\nF1-Score:")
    print(f"  ├─ Macro:    {test_metrics['f1_macro']:.4f}")
    print(f"  ├─ Micro:    {test_metrics['f1_micro']:.4f}")
    print(f"  └─ Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"\nRecall (Sensitivity):")
    print(f"  ├─ Macro:    {test_metrics['recall_macro']:.4f}")
    print(f"  ├─ Micro:    {test_metrics['recall_micro']:.4f}")
    print(f"  └─ Weighted: {test_metrics['recall_weighted']:.4f}")
    print(f"\nPrecision:")
    print(f"  ├─ Macro:    {test_metrics['precision_macro']:.4f}")
    print(f"  ├─ Micro:    {test_metrics['precision_micro']:.4f}")
    print(f"  └─ Weighted: {test_metrics['precision_weighted']:.4f}")
    print(f"\nSensitivity:")
    print(f"  ├─ Macro:    {test_metrics['sensitivity_macro']:.4f}")
    print(f"  ├─ Micro:    {test_metrics['sensitivity_micro']:.4f}")
    print(f"  └─ Weighted: {test_metrics['sensitivity_weighted']:.4f}")
    print(f"\nSpecificity:")
    print(f"  ├─ Macro:    {test_metrics['specificity_macro']:.4f}")
    print(f"  ├─ Micro:    {test_metrics['specificity_micro']:.4f}")
    print(f"  └─ Weighted: {test_metrics['specificity_weighted']:.4f}")
    print(f"\nKappa Score: {test_metrics['kappa']:.4f}")
    print(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}")
    print(f"Brier Score: {test_metrics['brier_score']:.4f}")

    # --- Plots ---
    print("\n=== Generating Plots ===")
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    roc_curves_path       = os.path.join(output_dir, 'roc_curves_per_class.png')
    pr_curves_path        = os.path.join(output_dir, 'pr_curves_per_class.png')
    overall_roc_path      = os.path.join(output_dir, 'overall_roc_curve.png')
    overall_pr_path       = os.path.join(output_dir, 'overall_pr_curve.png')
    training_curves_path  = os.path.join(output_dir, 'training_curves.png')
    convergence_rate_path = os.path.join(output_dir, 'convergence_rate.png')

    plot_confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'], NUM_CLASSES,
                          save_path=confusion_matrix_path)
    plot_roc_curves(test_metrics['y_true'], test_metrics['y_probs'], NUM_CLASSES,
                    save_path=roc_curves_path)
    plot_pr_curves(test_metrics['y_true'], test_metrics['y_probs'], NUM_CLASSES,
                   save_path=pr_curves_path)
    plot_overall_roc_curve(test_metrics['y_true'], test_metrics['y_probs'], NUM_CLASSES,
                           save_path=overall_roc_path)
    plot_overall_pr_curve(test_metrics['y_true'], test_metrics['y_probs'], NUM_CLASSES,
                          save_path=overall_pr_path)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch,
                         save_path=training_curves_path)
    plot_convergence_rate(train_losses, val_accs, save_path=convergence_rate_path)

    # --- Summary ---
    end_time = time.time()
    elapsed  = end_time - start_time
    hours    = int(elapsed // 3600)
    minutes  = int((elapsed % 3600) // 60)
    seconds  = int(elapsed % 60)

    print("\n=== Summary ===")
    print(f"Centralized QuanvTinyViT on ISIC2019_quanv:")
    print(f"  Model saved: {best_model_path}")
    print(f"\n  Final Test Metrics:")
    print(f"  ├─ Accuracy:               {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  ├─ Accuracy (Micro):       {test_metrics['accuracy_micro']:.4f}")
    print(f"  ├─ Accuracy (Macro):       {test_metrics['accuracy_macro']:.4f}")
    print(f"  ├─ AUC-ROC (Weighted):     {test_metrics['auc_weighted']:.4f}")
    print(f"  ├─ AUC-ROC (Macro):        {test_metrics['auc_macro']:.4f}")
    print(f"  ├─ F1 (Weighted):          {test_metrics['f1_weighted']:.4f}")
    print(f"  ├─ Recall (Weighted):      {test_metrics['recall_weighted']:.4f}")
    print(f"  ├─ Precision (Weighted):   {test_metrics['precision_weighted']:.4f}")
    print(f"  ├─ Kappa Score:            {test_metrics['kappa']:.4f}")
    print(f"  ├─ Matthews CC:            {test_metrics['mcc']:.4f}")
    print(f"  ├─ Sensitivity (Weighted): {test_metrics['sensitivity_weighted']:.4f}")
    print(f"  ├─ Specificity (Weighted): {test_metrics['specificity_weighted']:.4f}")
    print(f"  ├─ Brier Score:            {test_metrics['brier_score']:.4f}")
    print(f"  ├─ Total Params:           {complexity['total_params']:,} ({complexity['total_params']/1e6:.3f}M)")
    print(f"  ├─ Trainable Params:       {complexity['trainable_params']:,} ({complexity['trainable_params']/1e6:.3f}M)")
    print(f"  └─ FLOPs:                  {complexity['flops_str']}")
    print(f"\n  Saved artifacts:")
    print(f"  ├─ {confusion_matrix_path}")
    print(f"  ├─ {roc_curves_path}")
    print(f"  ├─ {pr_curves_path}")
    print(f"  ├─ {overall_roc_path}")
    print(f"  ├─ {overall_pr_path}")
    print(f"  ├─ {training_curves_path}")
    print(f"  └─ {convergence_rate_path}")
    print(f"\n  Total Time: {hours}h {minutes}m {seconds}s ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
