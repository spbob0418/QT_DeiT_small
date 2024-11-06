import torch
import numpy as np

def process_tensor(x):
    # x: Tensor of shape [BS, 197, 384]
    BS = x.shape[0]
    sequnece_length = x.shape[1]
    channel_dim = x.shape[2]
    num_elements = sequnece_length * channel_dim

    top1_values = []
    top3_values = []
    top1_percent_values = []
    top10_percent_values = []
    median_values = []

    for i in range(BS):
        sample = x[i]  # Shape [197, 384]
        flattened = sample.view(-1)  # Flatten to 1D tensor of size 197*384
        sorted_values, _ = torch.sort(flattened, descending=True)

        # Top1
        top1 = sorted_values[0].item()
        top1_values.append(top1)

        # Top3
        top3 = sorted_values[:3].numpy()
        top3_values.append(top3)

        # Top1% elements
        top1_percent_count = max(1, int(np.ceil(num_elements * 0.01)))
        top1_percent = sorted_values[:top1_percent_count].numpy()
        top1_percent_values.append(top1_percent)

        # Top10% elements
        top10_percent_count = max(1, int(np.ceil(num_elements * 0.10)))
        top10_percent = sorted_values[:top10_percent_count].numpy()
        top10_percent_values.append(top10_percent)

        # Median value
        median = torch.median(flattened).item()
        median_values.append(median)

    # Compute mean and standard deviation for each quantity
    top1_mean = np.mean(top1_values)
    top1_std = np.std(top1_values)

    top3_means = [np.mean(vals) for vals in top3_values]
    top3_mean = np.mean(top3_means)
    top3_std = np.std(top3_means)

    top1_percent_means = [np.mean(vals) for vals in top1_percent_values]
    top1_percent_mean = np.mean(top1_percent_means)
    top1_percent_std = np.std(top1_percent_means)

    top10_percent_means = [np.mean(vals) for vals in top10_percent_values]
    top10_percent_mean = np.mean(top10_percent_means)
    top10_percent_std = np.std(top10_percent_means)

    median_mean = np.mean(median_values)
    median_std = np.std(median_values)

    # Save results to a file
    with open('probe_result.txt', 'w') as f:
        f.write('Top1 Mean: {:.4f}, Top1 Std: {:.4f}\n'.format(top1_mean, top1_std))
        f.write('Top3 Mean: {:.4f}, Top3 Std: {:.4f}\n'.format(top3_mean, top3_std))
        f.write('Top1% Mean: {:.4f}, Top1% Std: {:.4f}\n'.format(top1_percent_mean, top1_percent_std))
        f.write('Top10% Mean: {:.4f}, Top10% Std: {:.4f}\n'.format(top10_percent_mean, top10_percent_std))
        f.write('Median Mean: {:.4f}, Median Std: {:.4f}\n'.format(median_mean, median_std))

    print("Results have been saved to 'probe_result.txt'.")

    # Visualization suggestion
    print("\nVisualization Suggestion:")
    print("You can create bar charts or box plots for each metric to visualize the mean and standard deviation.")
    print("For example, use matplotlib to plot the means with error bars representing the standard deviations.")

# Example usage:
# x = torch.randn(256, 197, 384)
# process_tensor(x)
