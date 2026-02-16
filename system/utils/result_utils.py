import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    print("std for best accuracy:", np.std(max_accuracy))
    print("mean for best accuracy:", np.mean(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
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
                    # Handle nested groups (like roc_curve and pr_curve)
                    if isinstance(round_group[key], h5py.Group):
                        results['detailed_metrics'][round_name][key] = {}
                        curve_group = round_group[key]
                        for curve_key in curve_group.keys():
                            results['detailed_metrics'][round_name][key][curve_key] = np.array(curve_group[curve_key][()])
                    else:
                        # Regular dataset
                        results['detailed_metrics'][round_name][key] = np.array(round_group[key][()])
        
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
        
        if 'accuracy' in final_metrics:
            print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        if 'f1_macro' in final_metrics:
            print(f"F1-Score (Macro): {final_metrics['f1_macro']:.4f}")
        if 'f1_weighted' in final_metrics:
            print(f"F1-Score (Weighted): {final_metrics['f1_weighted']:.4f}")
        if 'precision_macro' in final_metrics:
            print(f"Precision (Macro): {final_metrics['precision_macro']:.4f}")
        if 'recall_macro' in final_metrics:
            print(f"Recall (Macro): {final_metrics['recall_macro']:.4f}")
        if 'sensitivity_macro' in final_metrics:
            print(f"Sensitivity (Macro): {final_metrics['sensitivity_macro']:.4f}")
        if 'specificity_macro' in final_metrics:
            print(f"Specificity (Macro): {final_metrics['specificity_macro']:.4f}")
        if 'cohen_kappa' in final_metrics:
            print(f"Kappa Score: {final_metrics['cohen_kappa']:.4f}")
        if 'matthews_cc' in final_metrics:
            print(f"Matthews Correlation Coefficient: {final_metrics['matthews_cc']:.4f}")
        if 'auc_roc' in final_metrics:
            print(f"AUC-ROC: {final_metrics['auc_roc']:.4f}")
        if 'auc_pr' in final_metrics:
            print(f"AUC-PR: {final_metrics['auc_pr']:.4f}")
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
            print("\n--- QUANTUM MODEL INFO ---")
            model = results['fl_metrics']['model']
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


def plot_roc_curve(file_name, round_num=0, save_path=None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Args:
        file_name (str): Result file name (without .h5)
        round_num (int): Which round to plot (default: last round with data)
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
    if isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num
    
    # If round not found, use the last available round
    if round_key not in results['detailed_metrics']:
        available_rounds = list(results['detailed_metrics'].keys())
        if available_rounds:
            round_key = available_rounds[-1]
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


def plot_pr_curve(file_name, round_num=0, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        file_name (str): Result file name (without .h5)
        round_num (int): Which round to plot (default: last round with data)
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
    if isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num
    
    # If round not found, use the last available round
    if round_key not in results['detailed_metrics']:
        available_rounds = list(results['detailed_metrics'].keys())
        if available_rounds:
            round_key = available_rounds[-1]
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


def plot_roc_and_pr_curves(file_name, round_num=0, save_path=None):
    """
    Plot both ROC and PR curves side by side.
    
    Args:
        file_name (str): Result file name (without .h5)
        round_num (int): Which round to plot (default: last round with data)
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
        return
    
    # Find the round to use
    if isinstance(round_num, int):
        round_key = f'round_{round_num}'
    else:
        round_key = round_num
    
    if round_key not in results['detailed_metrics']:
        available_rounds = list(results['detailed_metrics'].keys())
        if available_rounds:
            round_key = available_rounds[-1]
        else:
            print("No rounds with curve data found")
            return
    
    metrics_dict = results['detailed_metrics'][round_key]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ROC curve
    if 'roc_curve' in metrics_dict and metrics_dict['roc_curve'] is not None:
        roc_curve_data = metrics_dict['roc_curve']
        if isinstance(roc_curve_data, dict) and 'fpr' in roc_curve_data and 'tpr' in roc_curve_data:
            fpr = roc_curve_data['fpr']
            tpr = roc_curve_data['tpr']
            axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {metrics_dict.get("auc_roc", 0):.4f})')
            axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            axes[0].set_xlabel('False Positive Rate', fontsize=11)
            axes[0].set_ylabel('True Positive Rate', fontsize=11)
            axes[0].set_title('ROC Curve', fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No ROC Curve Data', ha='center', va='center')
    
    # Plot PR curve
    if 'pr_curve' in metrics_dict and metrics_dict['pr_curve'] is not None:
        pr_curve_data = metrics_dict['pr_curve']
        if isinstance(pr_curve_data, dict) and 'precision' in pr_curve_data and 'recall' in pr_curve_data:
            precision = pr_curve_data['precision']
            recall = pr_curve_data['recall']
            axes[1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR (AUC = {metrics_dict.get("auc_pr", 0):.4f})')
            axes[1].set_xlabel('Recall', fontsize=11)
            axes[1].set_ylabel('Precision', fontsize=11)
            axes[1].set_title('Precision-Recall Curve', fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
    else:
        axes[1].text(0.5, 0.5, 'No PR Curve Data', ha='center', va='center')
    
    plt.suptitle(f'Model Performance Curves - {round_key}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curves saved to {save_path}")
    
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