"""
Comprehensive metrics calculation module for Federated Learning.

This module provides functions to calculate all evaluation metrics:
1. F1-Score
2. Accuracy
3. Recall
4. Precision
5. AUC-ROC Curve
6. PR-Curve
7. Kappa Score
8. Matthews Correlation Coefficient (MCC)
9. Confusion Matrix
10. Convergence Rate
11. Communication Overhead (per Round)
12. Computation Time (per Round)
"""

import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score,
    roc_auc_score, roc_curve, precision_recall_curve, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix
)


class MetricsCalculator:
    """Calculator for comprehensive evaluation metrics."""
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.metrics_dict = {}
    
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate all classification metrics for multi-class classification.
        
        Args:
            y_true (np.array): True labels (integers 0 to C-1), shape (N,)
            y_pred (np.array): Predicted labels (integers 0 to C-1), shape (N,)
            y_prob (np.array): Prediction probabilities from softmax, shape (N, C).
                               If None, ROC/PR curves won't be computed.
                               IMPORTANT: Must be softmax probabilities, not raw logits.
            
        Returns:
            dict: Dictionary containing all metric values
        """
        metrics_dict = {}
        
        # ===== Input Validation & Conversion =====
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        # Ensure integer types
        y_true = y_true.astype(int).flatten()
        y_pred = y_pred.astype(int).flatten()
        
        # Validate dimensions
        assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        assert len(y_true) > 0, "Empty predictions"
        
        # Validate label range
        assert np.min(y_true) >= 0 and np.max(y_true) < self.num_classes, \
            f"Labels out of range: min={np.min(y_true)}, max={np.max(y_true)}, expected [0, {self.num_classes-1}]"
        
        # print(f"[DEBUG] y_true: min={np.min(y_true)}, max={np.max(y_true)}, unique={np.unique(y_true)}")
        # print(f"[DEBUG] y_pred: min={np.min(y_pred)}, max={np.max(y_pred)}, unique={np.unique(y_pred)}")
        
        # ===== Basic Classification Metrics =====
        
        # 1. Accuracy
        metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        # In single-label multiclass, micro accuracy equals overall accuracy.
        metrics_dict['accuracy_micro'] = metrics_dict['accuracy']
        # print(f"[DEBUG] Accuracy: {metrics_dict['accuracy']:.4f}")
        
        # 2. Precision (macro, micro, weighted)
        metrics_dict['precision_macro'] = precision_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics_dict['precision_micro'] = precision_score(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics_dict['precision_weighted'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        # print(f"[DEBUG] Precision - macro: {metrics_dict['precision_macro']:.4f}, weighted: {metrics_dict['precision_weighted']:.4f}")
        
        # Per-class precision
        if self.num_classes > 2:
            precision_per_class = precision_score(
                y_true, y_pred, average=None, zero_division=0, labels=np.arange(self.num_classes)
            )
            metrics_dict['precision_per_class'] = precision_per_class
        
        # 3. Recall (macro, micro, weighted)
        metrics_dict['recall_macro'] = recall_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics_dict['recall_micro'] = recall_score(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics_dict['recall_weighted'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        # Balanced accuracy for multiclass corresponds to macro recall.
        metrics_dict['accuracy_macro'] = metrics_dict['recall_macro']
        # print(f"[DEBUG] Recall - macro: {metrics_dict['recall_macro']:.4f}, weighted: {metrics_dict['recall_weighted']:.4f}")
        
        # Per-class recall
        if self.num_classes > 2:
            recall_per_class = recall_score(
                y_true, y_pred, average=None, zero_division=0, labels=np.arange(self.num_classes)
            )
            metrics_dict['recall_per_class'] = recall_per_class
        
        # 4. F1-Score (macro, micro, weighted)
        metrics_dict['f1_macro'] = f1_score(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics_dict['f1_micro'] = f1_score(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics_dict['f1_weighted'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        # print(f"[DEBUG] F1 - macro: {metrics_dict['f1_macro']:.4f}, weighted: {metrics_dict['f1_weighted']:.4f}")
        
        # Per-class F1
        if self.num_classes > 2:
            f1_per_class = f1_score(
                y_true, y_pred, average=None, zero_division=0, labels=np.arange(self.num_classes)
            )
            metrics_dict['f1_per_class'] = f1_per_class
        
        # 5. Confusion Matrix
        metrics_dict['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))

        # 5b. Specificity (per class, macro, weighted)
        cm = metrics_dict['confusion_matrix']
        total = np.sum(cm)
        specificity_per_class = []
        supports = np.sum(cm, axis=1)
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = total - tp - fn - fp
            denom = tn + fp
            specificity_per_class.append((tn / denom) if denom > 0 else 0.0)

        specificity_per_class = np.array(specificity_per_class, dtype=np.float32)
        metrics_dict['specificity_per_class'] = specificity_per_class
        metrics_dict['specificity_macro'] = float(np.nanmean(specificity_per_class))
        if supports.sum() > 0:
            metrics_dict['specificity_weighted'] = float(
                np.average(specificity_per_class, weights=supports)
            )
        else:
            metrics_dict['specificity_weighted'] = 0.0

        # Sensitivity is recall (alias for clarity)
        metrics_dict['sensitivity_macro'] = metrics_dict['recall_macro']
        metrics_dict['sensitivity_micro'] = metrics_dict['recall_micro']
        metrics_dict['sensitivity_weighted'] = metrics_dict['recall_weighted']
        
        # 5c. Specificity - Micro average
        # Micro specificity: Overall TN / (Overall TN + Overall FP)
        total_tn = 0
        total_fp = 0
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = total - tp - fn - fp
            total_tn += tn
            total_fp += fp
        
        specificity_micro_denom = total_tn + total_fp
        metrics_dict['specificity_micro'] = float(total_tn / specificity_micro_denom) if specificity_micro_denom > 0 else 0.0
        
        # 6. Kappa Score
        metrics_dict['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # 7. Matthews Correlation Coefficient
        try:
            metrics_dict['matthews_cc'] = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            # print(f"[DEBUG] Could not compute Matthews CC: {e}")
            metrics_dict['matthews_cc'] = 0.0
        
        # ===== AUC and ROC/PR Curves (requires probability scores) =====
        
        if y_prob is not None:
            # Validate probability shape
            assert y_prob.shape[0] == len(y_true), \
                f"Probability shape mismatch: {y_prob.shape[0]} samples vs {len(y_true)} labels"
            assert y_prob.shape[1] == self.num_classes, \
                f"Probability class mismatch: {y_prob.shape[1]} classes vs {self.num_classes} expected"
            
            # print(f"[DEBUG] y_prob shape: {y_prob.shape}, min={np.min(y_prob):.4f}, max={np.max(y_prob):.4f}, sum per row: {np.mean(np.sum(y_prob, axis=1)):.4f}")
            
            # ===== Multiclass AUC-ROC =====
            metrics_dict['auc_roc'] = self._compute_multiclass_auc_roc(
                y_true, y_prob, self.num_classes
            )
            # print(f"[DEBUG] AUC-ROC: {metrics_dict['auc_roc']:.4f}")
            
            # ===== Multiclass AUC-PR =====
            metrics_dict['auc_pr'] = self._compute_multiclass_auc_pr(
                y_true, y_prob, self.num_classes
            )
            # print(f"[DEBUG] AUC-PR: {metrics_dict['auc_pr']:.4f}")
            
            # ===== ROC Curve (for class 1 in one-vs-rest) =====
            metrics_dict['roc_curve'] = self._compute_roc_curve(
                y_true, y_prob, self.num_classes
            )
            
            # ===== PR Curve (for class 1 in one-vs-rest) =====
            metrics_dict['pr_curve'] = self._compute_pr_curve(
                y_true, y_prob, self.num_classes
            )
        else:
            metrics_dict['auc_roc'] = 0.0
            metrics_dict['auc_pr'] = 0.0
            metrics_dict['roc_curve'] = None
            metrics_dict['pr_curve'] = None
        
        self.metrics_dict = metrics_dict
        return metrics_dict
    
    def _compute_multiclass_auc_roc(self, y_true, y_prob, num_classes):
        """
        Compute multiclass AUC-ROC using one-vs-rest approach.
        Robust to cases where some classes aren't predicted or aren't in ground truth.
        
        Args:
            y_true: True labels (0 to C-1)
            y_prob: Softmax probabilities (N, C)
            num_classes: Number of classes
            
        Returns:
            float: AUC-ROC score
        """
        try:
            # Check if we have more than one unique class
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                # print(f"[DEBUG] Warning: Only {len(unique_classes)} unique class(es) in ground truth")
                return 0.0
            
            # Binarize labels for one-vs-rest encoding
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            
            # Try weighted average first (handles imbalanced classes better)
            try:
                auc_roc = roc_auc_score(
                    y_true_bin,
                    y_prob,
                    average='weighted',
                    multi_class='ovr'
                )
                
                # If successful, return it
                if not (np.isnan(auc_roc) or np.isinf(auc_roc)):
                    return float(auc_roc)
            except Exception as e:
                print(f"[DEBUG] Weighted AUC failed: {e}, trying fallback...")
            
            # Fallback: macro average (treats all classes equally)
            try:
                auc_roc = roc_auc_score(
                    y_true_bin,
                    y_prob,
                    average='macro',
                    multi_class='ovr'
                )
                
                if not (np.isnan(auc_roc) or np.isinf(auc_roc)):
                    return float(auc_roc)
            except Exception as e:
                print(f"[DEBUG] Macro AUC failed: {e}, computing per-class...")
            
            # Final fallback: compute per-class and average manually
            try:
                auc_scores = []
                for class_idx in range(num_classes):
                    try:
                        # One-vs-rest for this class
                        y_binary = (y_true == class_idx).astype(int)
                        
                        # Only compute if both classes are present
                        if np.sum(y_binary) > 0 and np.sum(1 - y_binary) > 0:
                            auc_class = roc_auc_score(y_binary, y_prob[:, class_idx])
                            if not (np.isnan(auc_class) or np.isinf(auc_class)):
                                auc_scores.append(auc_class)
                    except Exception:
                        # Skip this class if AUC can't be computed
                        continue
                
                if auc_scores:
                    auc_roc = np.mean(auc_scores)
                    # print(f"[DEBUG] Computed AUC from {len(auc_scores)}/{num_classes} classes")
                    return float(auc_roc)
                else:
                    # print(f"[DEBUG] Could not compute AUC for any class")
                    return 0.0
            
            except Exception as e:
                print(f"[DEBUG] Per-class fallback failed: {e}")
                return 0.0
            
        except Exception as e:
            # print(f"[DEBUG] Error computing AUC-ROC: {e}")
            return 0.0
    
    def _compute_multiclass_auc_pr(self, y_true, y_prob, num_classes):
        """
        Compute multiclass AUC-PR using one-vs-rest approach (macro average).
        """
        try:
            auc_pr = 0.0
            valid_count = 0
            
            # Compute PR curve for each class and average
            for class_idx in range(num_classes):
                y_true_binary = (y_true == class_idx).astype(int)
                
                # Only compute if class is present
                if np.sum(y_true_binary) > 0:
                    y_prob_class = y_prob[:, class_idx]
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_prob_class)
                    
                    # Compute AUC-PR using auc function
                    from sklearn.metrics import auc as sk_auc
                    auc_pr_class = sk_auc(recall_curve, precision_curve)
                    
                    if not (np.isnan(auc_pr_class) or np.isinf(auc_pr_class)):
                        auc_pr += auc_pr_class
                        valid_count += 1
            
            if valid_count > 0:
                auc_pr = auc_pr / valid_count
            else:
                auc_pr = 0.0
            
            return float(auc_pr)
            
        except Exception as e:
            print(f"[DEBUG] Error computing AUC-PR: {e}")
            return 0.0
    
    def _compute_roc_curve(self, y_true, y_prob, num_classes):
        """
        Compute ROC curve for one-vs-rest (class 1 vs rest).
        """
        try:
            if num_classes < 2 or y_prob.shape[1] < 2:
                return None
            
            # Use class 1 (middle class or first non-zero class)
            target_class = min(1, num_classes - 1)
            y_true_binary = (y_true == target_class).astype(int)
            y_prob_class = y_prob[:, target_class]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
            
            return {
                'fpr': np.array(fpr, dtype=np.float32),
                'tpr': np.array(tpr, dtype=np.float32)
            }
        except Exception as e:
            print(f"[DEBUG] Error computing ROC curve: {e}")
            return None
    
    def _compute_pr_curve(self, y_true, y_prob, num_classes):
        """
        Compute PR curve for one-vs-rest (class 1 vs rest).
        """
        try:
            if num_classes < 2 or y_prob.shape[1] < 2:
                return None
            
            # Use class 1
            target_class = min(1, num_classes - 1)
            y_true_binary = (y_true == target_class).astype(int)
            y_prob_class = y_prob[:, target_class]
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_prob_class)
            
            return {
                'precision': np.array(precision_curve, dtype=np.float32),
                'recall': np.array(recall_curve, dtype=np.float32)
            }
        except Exception as e:
            print(f"[DEBUG] Error computing PR curve: {e}")
            return None


def auc_score(y_true, y_score):
    """Calculate AUC-PR score for binary classification."""
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
        from sklearn.metrics import auc
        return auc(recall_vals, precision_vals)
    except Exception:
        return 0.0


def auc_score_multiclass(y_true, y_score):
    """Calculate average AUC-PR score for multiclass."""
    try:
        from sklearn.metrics import auc
        auc_scores = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) > 0:
                precision_vals, recall_vals, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
                auc_scores.append(auc(recall_vals, precision_vals))
        return np.mean(auc_scores) if auc_scores else 0.0
    except Exception:
        return 0.0


class FLMetricsTracker:
    """Track federated learning specific metrics."""
    
    def __init__(self):
        self.round_times = []  # Time for each round
        self.communication_bytes = []  # Communication per round
        self.local_computation_times = []  # Computation time per round
        self.test_accuracies = []  # Test accuracy per round
        self.personalization_metrics = {}
        self.model_info = {}
    
    def add_round_time(self, time_cost):
        """Add time cost for a round."""
        self.round_times.append(time_cost)
    
    def add_communication_cost(self, bytes_transferred):
        """Add communication cost for a round (in bytes)."""
        self.communication_bytes.append(bytes_transferred)
    
    def add_local_computation_time(self, time_cost):
        """Add local computation time for a round."""
        self.local_computation_times.append(time_cost)
    
    def add_test_accuracy(self, accuracy):
        """Add test accuracy for a round."""
        self.test_accuracies.append(accuracy)

    def set_personalization_metrics(self, baseline_accuracy, personalized_accuracy):
        gain = personalized_accuracy - baseline_accuracy
        self.personalization_metrics = {
            'baseline_accuracy': baseline_accuracy,
            'personalized_accuracy': personalized_accuracy,
            'personalization_gain': gain
        }

    def set_model_info(self, model_info):
        if isinstance(model_info, dict):
            self.model_info = model_info
    
    def get_convergence_rate(self):
        """
        Calculate convergence rate as the difference between first and last accuracy.
        
        Returns:
            dict: Contains convergence rate metrics
        """
        if len(self.test_accuracies) < 2:
            return {
                'convergence_rate': 0.0,
                'initial_accuracy': 0.0,
                'final_accuracy': 0.0,
                'improvement': 0.0,
                'rounds_to_convergence': -1,
                'all_accuracies': self.test_accuracies
            }
        
        initial_acc = self.test_accuracies[0]
        final_acc = self.test_accuracies[-1]
        improvement = final_acc - initial_acc
        
        # Find rounds to convergence (95% of final accuracy)
        target_acc = initial_acc + 0.95 * improvement if improvement > 0 else final_acc
        rounds_to_convergence = -1
        for i, acc in enumerate(self.test_accuracies):
            if acc >= target_acc:
                rounds_to_convergence = i
                break
        
        return {
            'convergence_rate': improvement,
            'initial_accuracy': initial_acc,
            'final_accuracy': final_acc,
            'improvement': improvement,
            'rounds_to_convergence': rounds_to_convergence,
            'all_accuracies': self.test_accuracies
        }
    
    def get_communication_overhead(self):
        """Calculate communication overhead metrics."""
        if not self.communication_bytes:
            return {
                'total_communication_bytes': 0.0,
                'total_communication_mb': 0.0,
                'avg_communication_per_round': 0.0,
                'avg_communication_per_round_mb': 0.0,
                'communication_by_round': []
            }
        
        total_bytes = sum(self.communication_bytes)
        avg_per_round = np.mean(self.communication_bytes)
        
        return {
            'total_communication_bytes': total_bytes,
            'total_communication_mb': total_bytes / (1024 * 1024),
            'avg_communication_per_round': avg_per_round,
            'avg_communication_per_round_mb': avg_per_round / (1024 * 1024),
            'communication_by_round': self.communication_bytes
        }
    
    def get_computation_time(self):
        """Calculate computation time metrics."""
        if not self.round_times:
            return {
                'total_time_seconds': 0.0,
                'total_time_minutes': 0.0,
                'avg_time_per_round': 0.0,
                'std_time_per_round': 0.0,
                'min_time_per_round': 0.0,
                'max_time_per_round': 0.0,
                'time_by_round': []
            }
        
        total_time = sum(self.round_times)
        avg_per_round = np.mean(self.round_times)
        
        return {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'avg_time_per_round': avg_per_round,
            'std_time_per_round': np.std(self.round_times),
            'min_time_per_round': np.min(self.round_times),
            'max_time_per_round': np.max(self.round_times),
            'time_by_round': self.round_times
        }
    
    def get_all_fl_metrics(self):
        """Get all FL-specific metrics."""
        metrics = {
            'convergence': self.get_convergence_rate(),
            'communication': self.get_communication_overhead(),
            'computation': self.get_computation_time()
        }

        if self.personalization_metrics:
            metrics['personalization'] = self.personalization_metrics

        if self.model_info:
            metrics['model'] = self.model_info

        return metrics
