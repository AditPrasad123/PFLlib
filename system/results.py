from utils.result_utils import (
    plot_roc_and_pr_curves,
    print_detailed_metrics_summary,
    print_classwise_metrics,
    print_clientwise_metrics,
    extract_all_metrics_csv
)

file_name = 'ISIC2019_FedBABU_EfficientNetB0_test_0'

# Print overall comprehensive metrics
print_detailed_metrics_summary(file_name)

# Print class-wise metrics (per-class precision, recall, F1, support)
print_classwise_metrics(file_name)

# Print client-wise metrics (per-client accuracy, F1, precision, recall)
print_clientwise_metrics(file_name)

# Extract all metrics and save to CSV (optional)
# all_metrics = extract_all_metrics_csv(file_name, output_csv='metrics_summary.csv')

# Plot ROC and PR curves
# plot_roc_and_pr_curves(file_name)