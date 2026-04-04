import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def _compute_micro_sens_spec_from_confusion_matrix(cm):
    """Compute micro sensitivity/specificity from a multiclass confusion matrix."""
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return np.nan, np.nan

    tp_total = float(np.trace(cm))
    total = float(cm.sum())
    fn_total = total - tp_total
    fp_total = total - tp_total
    # One-vs-rest TN accumulated across classes.
    tn_total = float(cm.shape[0] * total - tp_total - fn_total - fp_total)

    sensitivity_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else np.nan
    specificity_micro = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else np.nan
    return sensitivity_micro, specificity_micro


def _compute_micro_macro_accuracy_from_confusion_matrix(cm):
    """Compute micro/macro accuracy from a multiclass confusion matrix."""
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return np.nan, np.nan

    total = float(cm.sum())
    accuracy_micro = float(np.trace(cm) / total) if total > 0 else np.nan

    row_sums = cm.sum(axis=1).astype(np.float64)
    per_class_recall = np.divide(
        np.diag(cm).astype(np.float64),
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0
    )
    accuracy_macro = float(np.mean(per_class_recall)) if per_class_recall.size > 0 else np.nan
    return accuracy_micro, accuracy_macro


def _fill_missing_micro_sens_spec(metrics_dict):
    """Fill missing micro sensitivity/specificity from confusion_matrix when available."""
    if not isinstance(metrics_dict, dict):
        return
    if 'confusion_matrix' not in metrics_dict:
        return

    needs_sens = 'sensitivity_micro' not in metrics_dict
    needs_spec = 'specificity_micro' not in metrics_dict
    if not (needs_sens or needs_spec):
        return

    sensitivity_micro, specificity_micro = _compute_micro_sens_spec_from_confusion_matrix(metrics_dict['confusion_matrix'])
    if needs_sens and not np.isnan(sensitivity_micro):
        metrics_dict['sensitivity_micro'] = float(sensitivity_micro)
    if needs_spec and not np.isnan(specificity_micro):
        metrics_dict['specificity_micro'] = float(specificity_micro)


def _fill_missing_micro_macro_accuracy(metrics_dict):
    """Fill missing micro/macro accuracy from confusion_matrix when available."""
    if not isinstance(metrics_dict, dict):
        return
    if 'confusion_matrix' not in metrics_dict:
        return

    accuracy_micro, accuracy_macro = _compute_micro_macro_accuracy_from_confusion_matrix(metrics_dict['confusion_matrix'])
    # Always synchronize with confusion matrix when available to avoid drift
    # from unweighted client-level averaging in older result files.
    if not np.isnan(accuracy_micro):
        metrics_dict['accuracy_micro'] = float(accuracy_micro)
        metrics_dict['accuracy'] = float(accuracy_micro)
    if not np.isnan(accuracy_macro):
        metrics_dict['accuracy_macro'] = float(accuracy_macro)


def average_data(algorithm="", dataset="", goal="", times=10, model="", prev=0):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, model, prev)

    max_accuracy = []
    for i in range(len(test_acc)):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, model="", prev=0):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(prev, times):
        if model:
            file_name = dataset + "_" + algorithm + "_" + model + "_" + goal + "_" + str(i)
        else:
            file_name = dataset + "_" + algorithm + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


def read_detailed_results(file_name):
    """
    Read all metrics from the results file including detailed metrics.
    
    Args:
        file_name (str): Name of the result file (without .h5 extension)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    file_path = "../results/" + file_name + ".h5"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}
    
    results = {}
    
    with h5py.File(file_path, 'r') as hf:
        # Read basic metrics
        if 'rs_test_acc' in hf:
            results['rs_test_acc'] = np.array(hf.get('rs_test_acc'))
        if 'rs_test_auc' in hf:
            results['rs_test_auc'] = np.array(hf.get('rs_test_auc'))
        if 'rs_train_loss' in hf:
            results['rs_train_loss'] = np.array(hf.get('rs_train_loss'))
        
        # Read detailed metrics
        if 'detailed_metrics' in hf:
            results['detailed_metrics'] = {}
            detailed_group = hf['detailed_metrics']
            for round_name in detailed_group.keys():
                results['detailed_metrics'][round_name] = {}
                round_group = detailed_group[round_name]
                for key in round_group.keys():
                    if isinstance(round_group[key], h5py.Group):
                        grp = round_group[key]
                        # Check for 2-level nesting (e.g., client_roc_curves/client_N/fpr)
                        if any(isinstance(grp[k], h5py.Group) for k in grp.keys()):
                            results['detailed_metrics'][round_name][key] = {}
                            for sub_key in grp.keys():
                                if isinstance(grp[sub_key], h5py.Group):
                                    results['detailed_metrics'][round_name][key][sub_key] = {}
                                    for leaf_key in grp[sub_key].keys():
                                        results['detailed_metrics'][round_name][key][sub_key][leaf_key] = np.array(grp[sub_key][leaf_key][()])
                                else:
                                    results['detailed_metrics'][round_name][key][sub_key] = np.array(grp[sub_key][()])
                        else:
                            # Single-level group (roc_curve, pr_curve)
                            results['detailed_metrics'][round_name][key] = {}
                            for curve_key in grp.keys():
                                results['detailed_metrics'][round_name][key][curve_key] = np.array(grp[curve_key][()])
                    else:
                        # Regular dataset
                        results['detailed_metrics'][round_name][key] = np.array(round_group[key][()])

            # Backfill missing micro metrics from confusion matrix for older result files.
            for round_name in results['detailed_metrics']:
                _fill_missing_micro_macro_accuracy(results['detailed_metrics'][round_name])
                _fill_missing_micro_sens_spec(results['detailed_metrics'][round_name])
        
        # Read FL-specific metrics
        if 'fl_metrics' in hf:
            results['fl_metrics'] = {}
            fl_group = hf['fl_metrics']
            
            if 'convergence' in fl_group:
                results['fl_metrics']['convergence'] = {}
                conv_group = fl_group['convergence']
                for key in conv_group.keys():
                    results['fl_metrics']['convergence'][key] = np.array(conv_group[key][()])
            
            if 'communication' in fl_group:
                results['fl_metrics']['communication'] = {}
                comm_group = fl_group['communication']
                for key in comm_group.keys():
                    results['fl_metrics']['communication'][key] = np.array(comm_group[key][()])
            
            if 'computation' in fl_group:
                results['fl_metrics']['computation'] = {}
                comp_group = fl_group['computation']
                for key in comp_group.keys():
                    results['fl_metrics']['computation'][key] = np.array(comp_group[key][()])

            if 'personalization' in fl_group:
                results['fl_metrics']['personalization'] = {}
                pers_group = fl_group['personalization']
                for key in pers_group.keys():
                    results['fl_metrics']['personalization'][key] = np.array(pers_group[key][()])

            if 'model' in fl_group:
                results['fl_metrics']['model'] = {}
                model_group = fl_group['model']
                for key in model_group.keys():
                    results['fl_metrics']['model'][key] = np.array(model_group[key][()])
    
    return results


def print_detailed_metrics_summary(file_name):
    """
    Print a comprehensive summary of all metrics from a results file.
    
    Args:
        file_name (str): Name of the result file (without .h5 extension)
    """
    results = read_detailed_results(file_name)
    
    if not results:
        return
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION METRICS SUMMARY")
    print("="*70)
    
    # Basic metrics
    print("\n--- BASIC METRICS ---")
    if 'rs_test_acc' in results and len(results['rs_test_acc']) > 0:
        print(f"Final Test Accuracy: {results['rs_test_acc'][-1]:.4f}")
        print(f"Best Test Accuracy: {np.max(results['rs_test_acc']):.4f}")
        print(f"Average Test Accuracy: {np.mean(results['rs_test_acc']):.4f}")
    
    if 'rs_test_auc' in results and len(results['rs_test_auc']) > 0:
        print(f"Final Test AUC: {results['rs_test_auc'][-1]:.4f}")
    
    if 'rs_train_loss' in results and len(results['rs_train_loss']) > 0:
        print(f"Final Train Loss: {results['rs_train_loss'][-1]:.4f}")
    
    # Detailed metrics
    if 'detailed_metrics' in results and len(results['detailed_metrics']) > 0:
        print("\n--- DETAILED CLASSIFICATION METRICS (Final Round) ---")
        final_round_idx = max(results['detailed_metrics'].keys(), 
                            key=lambda x: int(x.split('_')[1]))
        final_metrics = results['detailed_metrics'][final_round_idx]
        
        print("\n--- Accuracy (Macro, Micro, Overall) ---")
        if 'accuracy_macro' in final_metrics:
            print(f"  Macro: {final_metrics['accuracy_macro']:.4f}")
        if 'accuracy_micro' in final_metrics:
            print(f"  Micro: {final_metrics['accuracy_micro']:.4f}")
        if 'accuracy' in final_metrics:
            print(f"  Overall: {final_metrics['accuracy']:.4f}")
        
        print("\n--- Precision (Macro, Micro, Weighted) ---")
        if 'precision_macro' in final_metrics:
            print(f"  Macro: {final_metrics['precision_macro']:.4f}")
        if 'precision_micro' in final_metrics:
            print(f"  Micro: {final_metrics['precision_micro']:.4f}")
        if 'precision_weighted' in final_metrics:
            print(f"  Weighted: {final_metrics['precision_weighted']:.4f}")
        
        print("\n--- Recall (Macro, Micro, Weighted) ---")
        if 'recall_macro' in final_metrics:
            print(f"  Macro: {final_metrics['recall_macro']:.4f}")
        if 'recall_micro' in final_metrics:
            print(f"  Micro: {final_metrics['recall_micro']:.4f}")
        if 'recall_weighted' in final_metrics:
            print(f"  Weighted: {final_metrics['recall_weighted']:.4f}")
        
        print("\n--- Sensitivity (Macro, Micro, Weighted) ---")
        if 'sensitivity_macro' in final_metrics:
            print(f"  Macro: {final_metrics['sensitivity_macro']:.4f}")
        if 'sensitivity_micro' in final_metrics:
            print(f"  Micro: {final_metrics['sensitivity_micro']:.4f}")
        if 'sensitivity_weighted' in final_metrics:
            print(f"  Weighted: {final_metrics['sensitivity_weighted']:.4f}")
        
        print("\n--- Specificity (Macro, Micro, Weighted) ---")
        if 'specificity_macro' in final_metrics:
            print(f"  Macro: {final_metrics['specificity_macro']:.4f}")
        if 'specificity_micro' in final_metrics:
            print(f"  Micro: {final_metrics['specificity_micro']:.4f}")
        if 'specificity_weighted' in final_metrics:
            print(f"  Weighted: {final_metrics['specificity_weighted']:.4f}")
        
        print("\n--- F1-Score (Macro, Micro, Weighted) ---")
        if 'f1_macro' in final_metrics:
            print(f"  Macro: {final_metrics['f1_macro']:.4f}")
        if 'f1_micro' in final_metrics:
            print(f"  Micro: {final_metrics['f1_micro']:.4f}")
        if 'f1_weighted' in final_metrics:
            print(f"  Weighted: {final_metrics['f1_weighted']:.4f}")
        
        print("\n--- Other Metrics ---")
        if 'cohen_kappa' in final_metrics:
            print(f"Cohen's Kappa: {final_metrics['cohen_kappa']:.4f}")
        if 'matthews_cc' in final_metrics:
            print(f"Matthews Correlation Coefficient: {final_metrics['matthews_cc']:.4f}")
        if 'auc_roc' in final_metrics:
            print(f"AUC-ROC: {final_metrics['auc_roc']:.4f}")
        if 'auc_pr' in final_metrics:
            print(f"AUC-PR: {final_metrics['auc_pr']:.4f}")
        if 'brier_score' in final_metrics:
            try:
                bs = float(final_metrics['brier_score'])
                if not np.isnan(bs):
                    print(f"Brier Score: {bs:.4f}")
            except (TypeError, ValueError):
                pass
        if 'client_accuracy_variance' in final_metrics:
            print(f"Client Accuracy Variance: {final_metrics['client_accuracy_variance']:.6f}")
    else:
        print("\n⚠️  No detailed metrics found in this results file.")
        print("\n📝 To collect comprehensive metrics (F1, Precision, Recall, ROC/PR curves, etc.):")
        print("   Re-run training with the current code - metrics are collected automatically!")
        print("\n   Example command:")
        print("   cd system")
        print("   python main.py -data ISIC2019_quanv -m QuanvTinyViT -algo FedBABU \\")
        print("     -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda")
    
    # FL metrics
    if 'fl_metrics' in results:
        # Convergence
        if 'convergence' in results['fl_metrics']:
            print("\n--- CONVERGENCE METRICS ---")
            conv = results['fl_metrics']['convergence']
            if 'final_accuracy' in conv:
                print(f"Final Accuracy: {conv['final_accuracy']:.4f}")
            if 'initial_accuracy' in conv:
                print(f"Initial Accuracy: {conv['initial_accuracy']:.4f}")
            if 'improvement' in conv:
                print(f"Improvement: {conv['improvement']:.4f}")
            if 'rounds_to_convergence' in conv:
                rounds = conv['rounds_to_convergence']
                if rounds >= 0:
                    print(f"Rounds to 95% Convergence: {int(rounds)}")
        
        # Communication
        if 'communication' in results['fl_metrics']:
            print("\n--- COMMUNICATION OVERHEAD (Per Round) ---")
            comm = results['fl_metrics']['communication']
            if 'avg_communication_per_round_mb' in comm:
                print(f"Average per Round: {comm['avg_communication_per_round_mb']:.2f} MB")
            if 'total_communication_mb' in comm:
                print(f"Total: {comm['total_communication_mb']:.2f} MB")
        
        # Computation
        if 'computation' in results['fl_metrics']:
            print("\n--- COMPUTATION TIME (Per Round) ---")
            comp = results['fl_metrics']['computation']
            if 'avg_time_per_round' in comp:
                print(f"Average per Round: {comp['avg_time_per_round']:.4f} seconds")
            if 'total_time_minutes' in comp:
                print(f"Total: {comp['total_time_minutes']:.2f} minutes")
            if 'min_time_per_round' in comp:
                print(f"Min: {comp['min_time_per_round']:.4f} seconds")
            if 'max_time_per_round' in comp:
                print(f"Max: {comp['max_time_per_round']:.4f} seconds")

        if 'personalization' in results['fl_metrics']:
            print("\n--- PERSONALIZATION GAIN ---")
            pers = results['fl_metrics']['personalization']
            if 'baseline_accuracy' in pers:
                print(f"Baseline Accuracy: {float(pers['baseline_accuracy']):.4f}")
            if 'personalized_accuracy' in pers:
                print(f"Personalized Accuracy: {float(pers['personalized_accuracy']):.4f}")
            if 'personalization_gain' in pers:
                print(f"Gain: {float(pers['personalization_gain']):.4f}")

        if 'model' in results['fl_metrics']:
            print("\n--- MODEL INFO ---")
            model = results['fl_metrics']['model']
            if 'total_params' in model:
                print(f"Total Parameters: {int(model['total_params']):,}")
            if 'trainable_params' in model:
                print(f"Trainable Parameters: {int(model['trainable_params']):,}")
            if 'non_trainable_params' in model:
                print(f"Non-Trainable Parameters: {int(model['non_trainable_params']):,}")
            if 'flops_str' in model:
                flops_val = model['flops_str']
                if isinstance(flops_val, (bytes, np.bytes_)):
                    flops_val = flops_val.decode('utf-8')
                print(f"FLOPs: {flops_val}")
            if 'n_qubits' in model:
                print(f"Qubits: {int(model['n_qubits'])}")
            if 'circuit_depth' in model:
                print(f"Circuit Depth: {int(model['circuit_depth'])}")
    
    print("\n" + "="*70 + "\n")


def plot_convergence_curve(file_name, save_path=None):
    """
    Plot the convergence curve (accuracy vs rounds).
    
    Args:
        file_name (str): Name of the result file
        save_path (str): Path to save the plot (optional)
    """
    results = read_detailed_results(file_name)
    
    if 'rs_test_acc' not in results:
        print("No test accuracy data found")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['rs_test_acc'], linewidth=2, marker='o')
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Convergence Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_training_curves_fl(
    file_name,
    save_path=None,
    auto_clip_outlier=False,
    clip_percentile=95.0,
    hard_clip_threshold=None,
    hard_clip_value=None,
    hard_clip_rules=None,
):
    """
    Plot training loss and test accuracy side-by-side across federated rounds.

    Args:
        file_name (str): Name of the result file (without .h5 extension)
        save_path (str): Path to save the plot (optional)
        auto_clip_outlier (bool): If True, clip extreme loss spikes for readability.
        clip_percentile (float): Percentile used as clipping ceiling when auto clipping is enabled.
        hard_clip_threshold (float|None): If set, replace loss values > threshold.
        hard_clip_value (float|None): Replacement value used with hard_clip_threshold.
        hard_clip_rules (list[tuple[float, float]]|None): Ordered rules like
            [(10.0, 1.65), (2.1, 1.45)] interpreted as threshold bands.
    """
    results = read_detailed_results(file_name)

    has_loss = 'rs_train_loss' in results and len(results['rs_train_loss']) > 0
    has_acc  = 'rs_test_acc'   in results and len(results['rs_test_acc'])   > 0

    if not has_loss and not has_acc:
        print("No training loss or test accuracy data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {file_name}', fontsize=13, fontweight='bold')

    # Loss curve
    ax_loss = axes[0]
    if has_loss:
        loss_vals = np.array(results['rs_train_loss'], dtype=np.float64)
        rounds = np.arange(1, len(loss_vals) + 1)
        plot_vals = loss_vals.copy()

        clipped_rounds = []
        clip_ceiling = None
        hard_rule_applied = False

        # Explicit user-controlled clipping rule takes priority.
        if hard_clip_rules:
            # Apply non-overlapping threshold bands from high to low threshold.
            rules = sorted(
                [(float(t), float(v)) for (t, v) in hard_clip_rules],
                key=lambda x: x[0],
                reverse=True,
            )
            for idx, (thr, rep) in enumerate(rules):
                if idx == 0:
                    mask = loss_vals > thr
                else:
                    prev_thr = rules[idx - 1][0]
                    mask = (loss_vals > thr) & (loss_vals <= prev_thr)
                if np.any(mask):
                    plot_vals[mask] = rep
                    hard_rule_applied = True
            if hard_rule_applied:
                clipped_rounds = rounds[plot_vals != loss_vals].tolist()
        elif hard_clip_threshold is not None and hard_clip_value is not None:
            clipped_mask = loss_vals > float(hard_clip_threshold)
            if np.any(clipped_mask):
                clipped_rounds = rounds[clipped_mask].tolist()
                plot_vals[clipped_mask] = float(hard_clip_value)
                hard_rule_applied = True
        elif auto_clip_outlier and len(loss_vals) >= 4:
            clip_ceiling = float(np.percentile(loss_vals, clip_percentile))
            # Only clip when there is a clear spike beyond the chosen ceiling.
            if np.max(loss_vals) > clip_ceiling:
                clipped_mask = loss_vals > clip_ceiling
                clipped_rounds = rounds[clipped_mask].tolist()
                plot_vals = np.minimum(loss_vals, clip_ceiling)

        ax_loss.plot(rounds, plot_vals, color='royalblue', linewidth=2, marker='o', markersize=4, label='Train Loss')
        if clipped_rounds and (not hard_rule_applied):
            # Keep optional annotation only for percentile clipping mode.
            ax_loss.text(
                0.02,
                0.98,
                f'Clipped rounds: {clipped_rounds}\nRule: > p{clip_percentile:.0f}',
                transform=ax_loss.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.6),
            )
        ax_loss.set_xlabel('Round', fontsize=11)
        ax_loss.set_ylabel('Loss', fontsize=11)
        ax_loss.set_title('Training Loss vs Rounds', fontsize=12, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(fontsize=10)
    else:
        ax_loss.text(0.5, 0.5, 'No loss data', ha='center', va='center', fontsize=12)

    # Accuracy curve
    ax_acc = axes[1]
    if has_acc:
        acc_vals = results['rs_test_acc']
        rounds = np.arange(1, len(acc_vals) + 1)
        ax_acc.plot(rounds, acc_vals, color='darkorange', linewidth=2, marker='s', markersize=4, label='Test Accuracy')
        ax_acc.set_xlabel('Round', fontsize=11)
        ax_acc.set_ylabel('Accuracy', fontsize=11)
        ax_acc.set_title('Test Accuracy vs Rounds', fontsize=12, fontweight='bold')
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(fontsize=10)
    else:
        ax_acc.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_convergence_rate_fl(file_name, save_path=None):
    """
    Plot per-round convergence rate (loss improvement bars) for federated learning.

    Args:
        file_name (str): Name of the result file (without .h5 extension)
        save_path (str): Path to save the plot (optional)
    """
    results = read_detailed_results(file_name)

    has_loss = 'rs_train_loss' in results and len(results['rs_train_loss']) > 1
    has_acc  = 'rs_test_acc'   in results and len(results['rs_test_acc'])   > 1

    if not has_loss and not has_acc:
        print("Need at least 2 rounds of data to plot convergence rate.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Per-Round Convergence Rate — {file_name}', fontsize=13, fontweight='bold')

    # Loss delta bars
    ax1 = axes[0]
    if has_loss:
        loss_vals = np.array(results['rs_train_loss'])
        deltas = np.diff(loss_vals)          # negative = improving
        rounds = np.arange(2, len(loss_vals) + 1)
        colors = ['green' if d < 0 else 'red' for d in deltas]
        ax1.bar(rounds, -deltas, color=colors, edgecolor='black', linewidth=0.5)
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_xlabel('Round', fontsize=11)
        ax1.set_ylabel('Loss Improvement (Δ)', fontsize=11)
        ax1.set_title('Per-Round Loss Improvement\n(green = improving, red = worsening)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, 'No loss data', ha='center', va='center', fontsize=12)

    # Accuracy delta bars
    ax2 = axes[1]
    if has_acc:
        acc_vals = np.array(results['rs_test_acc'])
        deltas = np.diff(acc_vals)           # positive = improving
        rounds = np.arange(2, len(acc_vals) + 1)
        colors = ['green' if d > 0 else 'red' for d in deltas]
        ax2.bar(rounds, deltas, color=colors, edgecolor='black', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Round', fontsize=11)
        ax2.set_ylabel('Accuracy Improvement (Δ)', fontsize=11)
        ax2.set_title('Per-Round Accuracy Improvement\n(green = improving, red = worsening)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Annotate total improvement
        total_improvement = float(acc_vals[-1]) - float(acc_vals[0])
        ax2.set_title(
            f'Per-Round Accuracy Improvement\n(total: {total_improvement:+.4f})',
            fontsize=11, fontweight='bold'
        )
    else:
        ax2.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence rate plot saved to {save_path}")

    plt.show()


def compare_metrics(file_names, metric_key='rs_test_acc'):
    """
    Compare a specific metric across multiple runs.

    Args:
        file_names (list): List of result file names
        metric_key (str): Metric key to compare (default: 'rs_test_acc')
    """
    plt.figure(figsize=(12, 6))
    
    for fname in file_names:
        results = read_detailed_results(fname)
        if metric_key in results:
            plt.plot(results[metric_key], marker='o', label=fname)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel(metric_key, fontsize=12)
    plt.title(f'Comparison: {metric_key}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_class_roc_pr_curves(file_name, round_num=-1, save_path=None):
    """Plot class-wise ROC and PR curves for the selected round."""
    results = read_detailed_results(file_name)

    if 'detailed_metrics' not in results or len(results['detailed_metrics']) == 0:
        print("No detailed metrics found in this results file.")
        return

    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    if round_num == -1 or (isinstance(round_num, int) and round_num < 0):
        round_key = available_rounds[-1]
    elif isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num

    if round_key not in results['detailed_metrics']:
        round_key = available_rounds[-1]
        print(f"Note: Specified round not found, using {round_key}")

    metrics_dict = results['detailed_metrics'][round_key]
    class_roc_curves = metrics_dict.get('class_roc_curves', {})
    class_pr_curves = metrics_dict.get('class_pr_curves', {})

    if not class_roc_curves and not class_pr_curves:
        print("No class-wise ROC/PR curve data found in this result file.")
        print("Please re-run training with the updated code to populate class-wise curves.")
        return

    class_auc_roc = metrics_dict.get('class_auc_roc_by_class', None)
    class_auc_pr = metrics_dict.get('class_auc_pr_by_class', None)
    if isinstance(class_auc_roc, np.ndarray):
        class_auc_roc = class_auc_roc.tolist()
    if isinstance(class_auc_pr, np.ndarray):
        class_auc_pr = class_auc_pr.tolist()

    roc_items = sorted(class_roc_curves.items(), key=lambda kv: int(kv[0].split('_')[1]) if '_' in kv[0] else 0)
    pr_items = sorted(class_pr_curves.items(), key=lambda kv: int(kv[0].split('_')[1]) if '_' in kv[0] else 0)
    n_classes = max(len(roc_items), len(pr_items), 1)
    cmap = plt.cm.get_cmap('tab20', n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Class-wise ROC/PR Curves - {round_key}', fontsize=16, fontweight='bold')

    # ROC per class
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    for i, (class_name, curve_data) in enumerate(roc_items):
        if not isinstance(curve_data, dict) or 'fpr' not in curve_data or 'tpr' not in curve_data:
            continue
        class_idx = int(class_name.split('_')[1]) if '_' in class_name else i
        label = f'Class {class_idx}'
        if isinstance(class_auc_roc, list) and class_idx < len(class_auc_roc):
            try:
                auc_val = float(class_auc_roc[class_idx])
                if not np.isnan(auc_val):
                    label += f' (AUC={auc_val:.3f})'
            except (TypeError, ValueError):
                pass
        ax_roc.plot(np.array(curve_data['fpr']), np.array(curve_data['tpr']), color=cmap(i), lw=2.2, label=label)

    ax_roc.set_title('ROC Curves per Class', fontsize=14, fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate', fontsize=13)
    ax_roc.set_ylabel('True Positive Rate', fontsize=13)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(fontsize=11, loc='lower right', ncol=2 if n_classes > 12 else 1)

    # PR per class
    ax_pr = axes[1]
    for i, (class_name, curve_data) in enumerate(pr_items):
        if not isinstance(curve_data, dict) or 'precision' not in curve_data or 'recall' not in curve_data:
            continue
        class_idx = int(class_name.split('_')[1]) if '_' in class_name else i
        label = f'Class {class_idx}'
        if isinstance(class_auc_pr, list) and class_idx < len(class_auc_pr):
            try:
                auc_val = float(class_auc_pr[class_idx])
                if not np.isnan(auc_val):
                    label += f' (AUC={auc_val:.3f})'
            except (TypeError, ValueError):
                pass
        ax_pr.plot(np.array(curve_data['recall']), np.array(curve_data['precision']), color=cmap(i), lw=2.2, label=label)

    ax_pr.set_title('PR Curves per Class', fontsize=14, fontweight='bold')
    ax_pr.set_xlabel('Recall', fontsize=13)
    ax_pr.set_ylabel('Precision', fontsize=13)
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(fontsize=11, loc='best', ncol=2 if n_classes > 12 else 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class-wise ROC/PR curves saved to {save_path}")
    plt.show()


def plot_roc_curve(file_name, round_num=-1, save_path=None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Args:
        file_name (str): Result file name (without .h5)
        round_num (int): Which round to plot (default: -1 for last round)
        save_path (str): Optional path to save the plot
    """
    results = read_detailed_results(file_name)
    
    if 'detailed_metrics' not in results:
        print("\n❌ No detailed metrics found in results file.")
        print("\nThis file was created before the comprehensive metrics system was added.")
        print("\n📝 To generate ROC/PR curves, please:")
        print("   1. Re-run your training with the updated code")
        print("   2. The metrics will be automatically collected")
        print("\n   Example:")
        print("   cd system")
        print("   python main.py -data ISIC2019_quanv -m QuanvTinyViT -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda")
        return
    
    # Find the round to use
    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    
    if round_num == -1 or (isinstance(round_num, int) and round_num < 0):
        # Use last round
        round_key = available_rounds[-1]
    elif isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num
    
    # If round not found, use the last available round
    if round_key not in results['detailed_metrics']:
        if available_rounds:
            round_key = available_rounds[-1]
            print(f"Note: Specified round not found, using {round_key}")
        else:
            print("No rounds with ROC curve data found")
            return
    
    metrics_dict = results['detailed_metrics'][round_key]
    
    if 'roc_curve' not in metrics_dict:
        print(f"No ROC curve data found in {round_key}")
        return
    
    roc_curve_data = metrics_dict['roc_curve']
    
    # Handle different data formats
    if isinstance(roc_curve_data, dict):
        if 'fpr' in roc_curve_data and 'tpr' in roc_curve_data:
            fpr = roc_curve_data['fpr']
            tpr = roc_curve_data['tpr']
        else:
            print("Invalid ROC curve format")
            return
    elif isinstance(roc_curve_data, np.ndarray) and roc_curve_data.dtype == object:
        # Might be stored as dict in numpy array
        try:
            roc_curve_dict = dict(roc_curve_data)
            fpr = roc_curve_dict.get('fpr', None)
            tpr = roc_curve_dict.get('tpr', None)
            if fpr is None or tpr is None:
                print("Could not extract fpr/tpr from ROC curve data")
                return
        except Exception:
            print("Could not parse ROC curve data")
            return
    else:
        print("Unsupported ROC curve data format")
        return
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics_dict.get("auc_roc", 0):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {round_key}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_pr_curve(file_name, round_num=-1, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        file_name (str): Result file name (without .h5)
        round_num (int): Which round to plot (default: -1 for last round)
        save_path (str): Optional path to save the plot
    """
    results = read_detailed_results(file_name)
    
    if 'detailed_metrics' not in results:
        print("\n❌ No detailed metrics found in results file.")
        print("\nThis file was created before the comprehensive metrics system was added.")
        print("\n📝 To generate ROC/PR curves, please:")
        print("   1. Re-run your training with the updated code")
        print("   2. The metrics will be automatically collected")
        print("\n   Example:")
        print("   cd system")
        print("   python main.py -data ISIC2019_quanv -m QuanvTinyViT -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda")
        return
    
    # Find the round to use
    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    
    if round_num == -1 or (isinstance(round_num, int) and round_num < 0):
        # Use last round
        round_key = available_rounds[-1]
    elif isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num
    
    # If round not found, use the last available round
    if round_key not in results['detailed_metrics']:
        if available_rounds:
            round_key = available_rounds[-1]
            print(f"Note: Specified round not found, using {round_key}")
        else:
            print("No rounds with PR curve data found")
            return
    
    metrics_dict = results['detailed_metrics'][round_key]
    
    if 'pr_curve' not in metrics_dict:
        print(f"No PR curve data found in {round_key}")
        return
    
    pr_curve_data = metrics_dict['pr_curve']
    
    # Handle different data formats
    if isinstance(pr_curve_data, dict):
        if 'precision' in pr_curve_data and 'recall' in pr_curve_data:
            precision = pr_curve_data['precision']
            recall = pr_curve_data['recall']
        else:
            print("Invalid PR curve format")
            return
    elif isinstance(pr_curve_data, np.ndarray) and pr_curve_data.dtype == object:
        try:
            pr_curve_dict = dict(pr_curve_data)
            precision = pr_curve_dict.get('precision', None)
            recall = pr_curve_dict.get('recall', None)
            if precision is None or recall is None:
                print("Could not extract precision/recall from PR curve data")
                return
        except Exception:
            print("Could not parse PR curve data")
            return
    else:
        print("Unsupported PR curve data format")
        return
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {metrics_dict.get("auc_pr", 0):.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {round_key}', fontsize=14)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    plt.show()


def plot_client_roc_pr_curves(file_name, round_num=-1, save_path=None):
    """Plot client-wise ROC and PR curves for a chosen evaluation round."""
    results = read_detailed_results(file_name)

    if 'detailed_metrics' not in results or len(results['detailed_metrics']) == 0:
        print("No detailed metrics found in this results file.")
        return

    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    if round_num == -1 or (isinstance(round_num, int) and round_num < 0):
        round_key = available_rounds[-1]
    elif isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num

    if round_key not in results['detailed_metrics']:
        round_key = available_rounds[-1]
        print(f"Note: Specified round not found, using {round_key}")

    metrics_dict = results['detailed_metrics'][round_key]
    client_roc_curves = metrics_dict.get('client_roc_curves', {})
    client_pr_curves = metrics_dict.get('client_pr_curves', {})

    if not client_roc_curves and not client_pr_curves:
        print("No client-wise ROC/PR curve data found in this result file.")
        print("This can happen for older .h5 files generated before client-curve saving was added.")
        return

    # Support dict-of-dicts (HDF5 loaded) and list-of-dicts formats.
    def to_client_items(curve_obj):
        if isinstance(curve_obj, dict):
            return sorted(curve_obj.items(), key=lambda kv: int(kv[0].split('_')[1]) if '_' in kv[0] else 0)
        if isinstance(curve_obj, list):
            out = []
            for i, c in enumerate(curve_obj):
                if c is not None:
                    out.append((f'client_{i}', c))
            return out
        return []

    roc_items = to_client_items(client_roc_curves)
    pr_items = to_client_items(client_pr_curves)

    if len(roc_items) == 0 and len(pr_items) == 0:
        print("No valid client-wise ROC/PR curve data found.")
        return

    client_auc_roc = metrics_dict.get('client_auc_roc_list', None)
    client_auc_pr = metrics_dict.get('client_auc_pr_list', None)
    if isinstance(client_auc_roc, np.ndarray):
        client_auc_roc = client_auc_roc.tolist()
    if isinstance(client_auc_pr, np.ndarray):
        client_auc_pr = client_auc_pr.tolist()

    n_clients = max(len(roc_items), len(pr_items), 1)
    cmap = plt.cm.get_cmap('tab20', n_clients)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Client-wise ROC/PR Curves - {round_key}', fontsize=16, fontweight='bold')

    # ROC panel
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    for i, (client_name, curve_data) in enumerate(roc_items):
        if not isinstance(curve_data, dict) or 'fpr' not in curve_data or 'tpr' not in curve_data:
            continue
        label = client_name
        if isinstance(client_auc_roc, list) and i < len(client_auc_roc):
            try:
                val = float(client_auc_roc[i])
                if not np.isnan(val):
                    label += f' (AUC={val:.3f})'
            except (TypeError, ValueError):
                pass
        ax_roc.plot(np.array(curve_data['fpr']), np.array(curve_data['tpr']), color=cmap(i), lw=2.2, label=label)

    ax_roc.set_title('ROC Curves per Client', fontsize=14, fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate', fontsize=13)
    ax_roc.set_ylabel('True Positive Rate', fontsize=13)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(fontsize=11, loc='lower right', ncol=2 if n_clients > 12 else 1)

    # PR panel
    ax_pr = axes[1]
    for i, (client_name, curve_data) in enumerate(pr_items):
        if not isinstance(curve_data, dict) or 'precision' not in curve_data or 'recall' not in curve_data:
            continue
        label = client_name
        if isinstance(client_auc_pr, list) and i < len(client_auc_pr):
            try:
                val = float(client_auc_pr[i])
                if not np.isnan(val):
                    label += f' (AUC={val:.3f})'
            except (TypeError, ValueError):
                pass
        ax_pr.plot(np.array(curve_data['recall']), np.array(curve_data['precision']), color=cmap(i), lw=2.2, label=label)

    ax_pr.set_title('PR Curves per Client', fontsize=14, fontweight='bold')
    ax_pr.set_xlabel('Recall', fontsize=13)
    ax_pr.set_ylabel('Precision', fontsize=13)
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(fontsize=11, loc='best', ncol=2 if n_clients > 12 else 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Client-wise ROC/PR curves saved to {save_path}")
    plt.show()


def plot_roc_and_pr_curves(file_name, round_num=-1, save_path=None):
    """
    Plot ROC, PR curves, and Confusion Matrix.
    
    Args:
        file_name (str): Result file name (without .h5)
        round_num (int): Which round to plot (default: -1 for last round)
        save_path (str): Optional path to save the plot
    """
    results = read_detailed_results(file_name)
    
    if 'detailed_metrics' not in results:
        print("\n❌ No detailed metrics found in results file.")
        print("\nThis file was created before the comprehensive metrics system was added.")
        print("\n📝 To generate ROC/PR curves, please:")
        print("   1. Re-run your training with the updated code")
        print("   2. The metrics will be automatically collected")
        print("\n   Example:")
        print("   cd system")
        print("   python main.py -data ISIC2019_quanv -m QuanvTinyViT -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda")
        return
    
    # Find the round to use
    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    
    if round_num == -1 or (isinstance(round_num, int) and round_num < 0):
        # Use last round
        round_key = available_rounds[-1]
    elif isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num
    
    # If round not found, use the last available round
    if round_key not in results['detailed_metrics']:
        if available_rounds:
            round_key = available_rounds[-1]
            print(f"Note: Specified round not found, using {round_key}")
        else:
            print("No rounds with curve data found")
            return
    
    metrics_dict = results['detailed_metrics'][round_key]
    
    # Create 2x2 subplot: ROC, PR, Confusion Matrix, and metrics summary
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ROC Curve (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'roc_curve' in metrics_dict and metrics_dict['roc_curve'] is not None:
        roc_curve_data = metrics_dict['roc_curve']
        if isinstance(roc_curve_data, dict) and 'fpr' in roc_curve_data and 'tpr' in roc_curve_data:
            fpr = roc_curve_data['fpr']
            tpr = roc_curve_data['tpr']
            ax1.plot(fpr, tpr, color='darkorange', lw=2.8, label=f'ROC (AUC = {metrics_dict.get("auc_roc", 0):.4f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax1.set_xlabel('False Positive Rate', fontsize=13)
            ax1.set_ylabel('True Positive Rate', fontsize=13)
            ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=12, loc='lower right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
    else:
        ax1.text(0.5, 0.5, 'No ROC Curve Data', ha='center', va='center', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
    
    # PR Curve (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'pr_curve' in metrics_dict and metrics_dict['pr_curve'] is not None:
        pr_curve_data = metrics_dict['pr_curve']
        if isinstance(pr_curve_data, dict) and 'precision' in pr_curve_data and 'recall' in pr_curve_data:
            precision = pr_curve_data['precision']
            recall = pr_curve_data['recall']
            ax2.plot(recall, precision, color='darkgreen', lw=2.8, label=f'PR (AUC = {metrics_dict.get("auc_pr", 0):.4f})')
            ax2.set_xlabel('Recall', fontsize=13)
            ax2.set_ylabel('Precision', fontsize=13)
            ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=12, loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
    else:
        ax2.text(0.5, 0.5, 'No PR Curve Data', ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    # Confusion Matrix (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'confusion_matrix' in metrics_dict and metrics_dict['confusion_matrix'] is not None:
        cm = np.array(metrics_dict['confusion_matrix'])
        
        # Create heatmap
        im = ax3.imshow(cm, cmap='Blues', aspect='auto')
        
        # Function to determine if text should be white or black based on background brightness
        def get_text_color(value, max_value):
            # Normalize value to 0-1 range
            normalized = value / max_value if max_value > 0 else 0
            # If normalized value is high (dark background), use white text, else black
            return 'white' if normalized > 0.5 else 'black'
        
        # Add text annotations with dynamic color
        num_classes = cm.shape[0]
        max_value = np.max(cm) if np.max(cm) > 0 else 1
        for i in range(num_classes):
            for j in range(num_classes):
                value = cm[i, j]
                text_color = get_text_color(value, max_value)
                text = ax3.text(j, i, int(value),
                              ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)
        
        ax3.set_xlabel('Predicted Label', fontsize=13)
        ax3.set_ylabel('True Label', fontsize=13)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax3.set_xticks(np.arange(num_classes))
        ax3.set_yticks(np.arange(num_classes))
        plt.colorbar(im, ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'No Confusion Matrix Data', ha='center', va='center', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # Metrics Summary (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary text
    summary_text = "=== METRICS SUMMARY ===\n\n"
    
    summary_text += "ACCURACY:\n"
    if 'accuracy_macro' in metrics_dict:
        summary_text += f"  Macro: {metrics_dict['accuracy_macro']:.4f}\n"
    if 'accuracy_micro' in metrics_dict:
        summary_text += f"  Micro: {metrics_dict['accuracy_micro']:.4f}\n"
    if 'accuracy' in metrics_dict:
        summary_text += f"  Overall: {metrics_dict['accuracy']:.4f}\n"
    
    summary_text += "\nPRECISION:\n"
    if 'precision_macro' in metrics_dict:
        summary_text += f"  Macro: {metrics_dict['precision_macro']:.4f}\n"
    if 'precision_micro' in metrics_dict:
        summary_text += f"  Micro: {metrics_dict['precision_micro']:.4f}\n"
    
    summary_text += "\nRECALL:\n"
    if 'recall_macro' in metrics_dict:
        summary_text += f"  Macro: {metrics_dict['recall_macro']:.4f}\n"
    if 'recall_micro' in metrics_dict:
        summary_text += f"  Micro: {metrics_dict['recall_micro']:.4f}\n"
    
    summary_text += "\nSENSITIVITY:\n"
    if 'sensitivity_macro' in metrics_dict:
        summary_text += f"  Macro: {metrics_dict['sensitivity_macro']:.4f}\n"
    if 'sensitivity_micro' in metrics_dict:
        summary_text += f"  Micro: {metrics_dict['sensitivity_micro']:.4f}\n"
    
    summary_text += "\nSPECIFICITY:\n"
    if 'specificity_macro' in metrics_dict:
        summary_text += f"  Macro: {metrics_dict['specificity_macro']:.4f}\n"
    if 'specificity_micro' in metrics_dict:
        summary_text += f"  Micro: {metrics_dict['specificity_micro']:.4f}\n"
    
    summary_text += "\nF1-SCORE:\n"
    if 'f1_macro' in metrics_dict:
        summary_text += f"  Macro: {metrics_dict['f1_macro']:.4f}\n"
    if 'f1_micro' in metrics_dict:
        summary_text += f"  Micro: {metrics_dict['f1_micro']:.4f}\n"
    
    summary_text += "\nOTHER:\n"
    if 'cohen_kappa' in metrics_dict:
        summary_text += f"  Kappa: {metrics_dict['cohen_kappa']:.4f}\n"
    if 'matthews_cc' in metrics_dict:
        summary_text += f"  MCC: {metrics_dict['matthews_cc']:.4f}\n"
    if 'brier_score' in metrics_dict:
        try:
            bs = float(metrics_dict['brier_score'])
            if not np.isnan(bs):
                summary_text += f"  Brier Score: {bs:.4f}\n"
        except (TypeError, ValueError):
            pass
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Model Performance Evaluation - {round_key}', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curves and confusion matrix saved to {save_path}")
    
    plt.show()


def plot_curves_across_rounds(file_name, save_path=None):
    """
    Plot ROC curves for multiple evaluation rounds in separate subplots.
    
    Args:
        file_name (str): Result file name (without .h5)
        save_path (str): Optional path to save the plot
    """
    results = read_detailed_results(file_name)
    
    if 'detailed_metrics' not in results:
        print("\n❌ No detailed metrics found in results file.")
        print("\nThis file was created before the comprehensive metrics system was added.")
        print("\n📝 To generate ROC/PR curves, please:")
        print("   1. Re-run your training with the updated code")
        print("   2. The metrics will be automatically collected")
        print("\n   Example:")
        print("   cd system")
        print("   python main.py -data ISIC2019_quanv -m QuanvTinyViT -algo FedBABU -gr 20 -ls 2 -lbs 16 -fb True -lr 0.001 -fte 10 -dev cuda")
        return
    
    # Get all rounds with ROC curve data
    rounds_with_roc = []
    for round_key, metrics_dict in results['detailed_metrics'].items():
        if 'roc_curve' in metrics_dict and metrics_dict['roc_curve'] is not None:
            rounds_with_roc.append((round_key, metrics_dict))
    
    if not rounds_with_roc:
        print("No rounds with ROC curve data found")
        return
    
    # Plot multiple rounds
    num_rounds = len(rounds_with_roc)
    num_cols = min(3, num_rounds)
    num_rows = (num_rounds + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(num_rows, num_cols)
    
    for idx, (round_key, metrics_dict) in enumerate(rounds_with_roc):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col]
        
        roc_curve_data = metrics_dict['roc_curve']
        if isinstance(roc_curve_data, dict) and 'fpr' in roc_curve_data:
            fpr = roc_curve_data['fpr']
            tpr = roc_curve_data['tpr']
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={metrics_dict.get("auc_roc", 0):.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel('FPR', fontsize=10)
            ax.set_ylabel('TPR', fontsize=10)
            ax.set_title(f'{round_key}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_rounds, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    
    plt.suptitle('ROC Curves Across Evaluation Rounds', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curves saved to {save_path}")
    
    plt.show()


# ==================== CLASS-WISE AND CLIENT-WISE METRICS ====================

def print_classwise_metrics(file_name, round_num=-1):
    """
    Extract and print per-class metrics (Precision, Recall, F1, Sensitivity, Specificity, Support)
    
    Args:
        file_name (str): Name of the result file (without .h5 extension)
        round_num (int): Round number to display (-1 for final round)
    """
    results = read_detailed_results(file_name)
    
    if 'detailed_metrics' not in results or len(results['detailed_metrics']) == 0:
        print("⚠️  No detailed metrics found in this results file.")
        return
    
    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    
    if round_num == -1:
        round_key = available_rounds[-1]
    else:
        round_key = f'round_{round_num}'
    
    if round_key not in results['detailed_metrics']:
        print(f"Round {round_key} not found in results.")
        return
    
    metrics_dict = results['detailed_metrics'][round_key]
    
    print("\n" + "="*80)
    print(f"CLASS-WISE METRICS ({round_key})")
    print("="*80)
    
    # Extract per-class metrics from confusion matrix if available
    if 'confusion_matrix' in metrics_dict:
        cm = metrics_dict['confusion_matrix']
        num_classes = cm.shape[0]
        
        print(f"\n{'Class':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 68)
        
        for class_id in range(num_classes):
            # Calculate metrics from confusion matrix
            tp = cm[class_id, class_id]
            fp = cm[:, class_id].sum() - tp
            fn = cm[class_id, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Per-class accuracy: (TP + TN) / Total
            accuracy = (tp + tn) / cm.sum() if cm.sum() > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = cm[class_id, :].sum()
            
            print(f"{class_id:<8} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {int(support):<10}")
    else:
        print("Confusion matrix not available in this results file.")


def print_clientwise_metrics(file_name, round_num=-1):
    """
    Extract and print per-client metrics (Accuracy, F1, Precision, Recall, Test Samples)
    
    Args:
        file_name (str): Name of the result file (without .h5 extension)  
        round_num (int): Round number to display (-1 for final round)
    """
    results = read_detailed_results(file_name)
    
    if 'detailed_metrics' not in results or len(results['detailed_metrics']) == 0:
        print("⚠️  No detailed metrics found in this results file.")
        return
    
    available_rounds = sorted(results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))
    
    if round_num == -1:
        round_key = available_rounds[-1]
    else:
        round_key = f'round_{round_num}'
    
    if round_key not in results['detailed_metrics']:
        print(f"Round {round_key} not found in results.")
        return
    
    metrics_dict = results['detailed_metrics'][round_key]
    
    print("\n" + "="*100)
    print(f"CLIENT-WISE METRICS ({round_key})")
    print("="*100)
    
    # Print per-client accuracy distribution
    if 'client_accuracy_by_client' in metrics_dict:
        client_accs = metrics_dict['client_accuracy_by_client']
        
        # Get per-client metric lists if available
        client_f1_list = metrics_dict.get('client_f1_macro_list', [np.nan] * len(client_accs))
        client_precision_list = metrics_dict.get('client_precision_macro_list', [np.nan] * len(client_accs))
        client_recall_list = metrics_dict.get('client_recall_macro_list', [np.nan] * len(client_accs))
        
        print(f"\n{'Client':<10} {'Accuracy':<15} {'F1-Macro':<15} {'Precision':<15} {'Recall':<15}")
        print("-" * 70)
        
        for client_id, acc in enumerate(client_accs):
            f1 = client_f1_list[client_id] if client_id < len(client_f1_list) else np.nan
            precision = client_precision_list[client_id] if client_id < len(client_precision_list) else np.nan
            recall = client_recall_list[client_id] if client_id < len(client_recall_list) else np.nan
            
            f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
            precision_str = f"{precision:.4f}" if not np.isnan(precision) else "N/A"
            recall_str = f"{recall:.4f}" if not np.isnan(recall) else "N/A"
            
            print(f"{client_id:<10} {acc:<15.4f} {f1_str:<15} {precision_str:<15} {recall_str:<15}")
        
        # Print statistics
        print("-" * 70)
        print(f"{'Mean':<10} {np.mean(client_accs):<15.4f}")
        print(f"{'Std Dev':<10} {np.std(client_accs):<15.4f}")
        print(f"{'Max':<10} {np.max(client_accs):<15.4f}")
        print(f"{'Min':<10} {np.min(client_accs):<15.4f}")
    else:
        print("Per-client metrics not available in this results file.")
        print("Note: Client-wise metrics are collected during federated training.")


def extract_all_metrics_csv(file_name, output_csv=None):
    """
    Extract all metrics (overall, class-wise, client-wise) and optionally save to CSV
    
    Args:
        file_name (str): Name of the result file (without .h5 extension)
        output_csv (str): Optional path to save metrics as CSV
        
    Returns:
        dict: Dictionary containing all extracted metrics
    """
    results = read_detailed_results(file_name)
    all_metrics = {}
    
    if 'detailed_metrics' not in results:
        print("No detailed metrics found.")
        return all_metrics
    
    for round_key, metrics_dict in results['detailed_metrics'].items():
        round_idx = int(round_key.split('_')[1])
        all_metrics[round_idx] = {}
        
        # Overall metrics
        all_metrics[round_idx]['overall'] = {
            'accuracy': metrics_dict.get('accuracy', np.nan),
            'accuracy_micro': metrics_dict.get('accuracy_micro', np.nan),
            'accuracy_macro': metrics_dict.get('accuracy_macro', np.nan),
            'f1_macro': metrics_dict.get('f1_macro', np.nan),
            'f1_micro': metrics_dict.get('f1_micro', np.nan),
            'f1_weighted': metrics_dict.get('f1_weighted', np.nan),
            'precision_macro': metrics_dict.get('precision_macro', np.nan),
            'precision_micro': metrics_dict.get('precision_micro', np.nan),
            'recall_macro': metrics_dict.get('recall_macro', np.nan),
            'recall_micro': metrics_dict.get('recall_micro', np.nan),
            'sensitivity_macro': metrics_dict.get('sensitivity_macro', np.nan),
            'sensitivity_micro': metrics_dict.get('sensitivity_micro', np.nan),
            'specificity_macro': metrics_dict.get('specificity_macro', np.nan),
            'specificity_micro': metrics_dict.get('specificity_micro', np.nan),
            'kappa': metrics_dict.get('cohen_kappa', np.nan),
            'mcc': metrics_dict.get('matthews_cc', np.nan),
            'auc_roc': metrics_dict.get('auc_roc', np.nan),
            'auc_pr': metrics_dict.get('auc_pr', np.nan),
            'brier_score': metrics_dict.get('brier_score', np.nan),
        }
        
        # Class-wise metrics from confusion matrix
        if 'confusion_matrix' in metrics_dict:
            cm = metrics_dict['confusion_matrix']
            num_classes = cm.shape[0]
            all_metrics[round_idx]['class_wise'] = {}
            
            for class_id in range(num_classes):
                tp = cm[class_id, class_id]
                fp = cm[:, class_id].sum() - tp
                fn = cm[class_id, :].sum() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                support = int(cm[class_id, :].sum())
                
                all_metrics[round_idx]['class_wise'][class_id] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                }
        
        # Client-wise metrics
        if 'client_accuracy_by_client' in metrics_dict:
            all_metrics[round_idx]['client_wise'] = {}
            client_accs = metrics_dict['client_accuracy_by_client']
            
            for client_id, acc in enumerate(client_accs):
                all_metrics[round_idx]['client_wise'][client_id] = {
                    'accuracy': acc
                }
    
    # Optionally save to CSV
    if output_csv:
        import pandas as pd
        csv_data = []
        
        for round_idx in sorted(all_metrics.keys()):
            metrics = all_metrics[round_idx]
            
            # Add overall metrics
            row = {'round': round_idx, 'metric_type': 'overall'}
            if 'overall' in metrics:
                row.update({f"overall_{k}": v for k, v in metrics['overall'].items()})
            csv_data.append(row)
            
            # Add class-wise metrics
            if 'class_wise' in metrics:
                for class_id, class_metrics in metrics['class_wise'].items():
                    row = {'round': round_idx, 'metric_type': f'class_{class_id}'}
                    row.update({f"class_{class_id}_{k}": v for k, v in class_metrics.items()})
                    csv_data.append(row)
            
            # Add client-wise metrics
            if 'client_wise' in metrics:
                for client_id, client_metrics in metrics['client_wise'].items():
                    row = {'round': round_idx, 'metric_type': f'client_{client_id}'}
                    row.update({f"client_{client_id}_{k}": v for k, v in client_metrics.items()})
                    csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv, index=False)
        print(f"\nMetrics saved to {output_csv}")
    
    return all_metrics