import os

from utils.result_utils import (
    plot_roc_and_pr_curves,
    plot_class_roc_pr_curves,
    read_detailed_results,
    print_detailed_metrics_summary,
    print_classwise_metrics,
    print_clientwise_metrics,
    extract_all_metrics_csv,
    plot_training_curves_fl,
    plot_convergence_rate_fl,
    plot_client_roc_pr_curves,
)

file_name = 'ISIC2019_quanv_FedAvg_QuanvEfficientNetB0_test_0'
output_dir = os.path.join('..', 'results', f'{file_name}_plots')
os.makedirs(output_dir, exist_ok=True)

# Quick check: verify class-wise curve keys exist in the selected result file.
_results = read_detailed_results(file_name)
if 'detailed_metrics' in _results and len(_results['detailed_metrics']) > 0:
    _last_round = sorted(_results['detailed_metrics'].keys(), key=lambda x: int(x.split('_')[1]))[-1]
    _m = _results['detailed_metrics'][_last_round]
    has_class_roc = 'class_roc_curves' in _m and bool(_m['class_roc_curves'])
    has_class_pr = 'class_pr_curves' in _m and bool(_m['class_pr_curves'])
    print(f"Class-wise ROC present: {has_class_roc}")
    print(f"Class-wise PR present: {has_class_pr}")
else:
    print("No detailed metrics found in selected file.")

# Print overall comprehensive metrics
print_detailed_metrics_summary(file_name)

# Print class-wise metrics (per-class precision, recall, F1, support)
print_classwise_metrics(file_name)

# Print client-wise metrics (per-client accuracy, F1, precision, recall)
print_clientwise_metrics(file_name)

# Extract all metrics and save to CSV (optional)
# all_metrics = extract_all_metrics_csv(file_name, output_csv='metrics_summary.csv')

# Plot ROC and PR curves
plot_roc_and_pr_curves(file_name, save_path=os.path.join(output_dir, 'roc_pr_confusion_summary.png'))

# Plot training loss and accuracy curves across federated rounds
plot_training_curves_fl(
    file_name,
    save_path=os.path.join(output_dir, 'training_curves.png'),
    hard_clip_threshold=3.0,
    hard_clip_value=0.2,
)

# Plot per-round convergence rate (loss/accuracy improvement bars)
plot_convergence_rate_fl(file_name, save_path=os.path.join(output_dir, 'convergence_rate.png'))

# Plot per-client ROC and PR curves (one line per client, coloured)
plot_client_roc_pr_curves(file_name, save_path=os.path.join(output_dir, 'client_roc_pr_curves.png'))

# Plot per-class ROC and PR curves (one line per class, coloured)
plot_class_roc_pr_curves(file_name, save_path=os.path.join(output_dir, 'class_roc_pr_curves.png'))

print(f"Saved plots to: {os.path.abspath(output_dir)}")