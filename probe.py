import torch
import numpy as np
import csv

BATCH256 = 1252
BATCH128 = 2502

def process_tensor(x, iteration, layer: str):
    # x: Tensor of shape [BS, sequence_length, channel_dim]
    BS = x.shape[0]
    sequence_length = x.shape[1]
    channel_dim = x.shape[2]
    num_elements = sequence_length * channel_dim

    top1_values = []
    top3_values = []
    top3_indices = []
    top1_percent_values = []
    median_values = []

    for i in range(BS):
        sample = x[i]  # Shape [sequence_length, channel_dim]
        flattened = sample.view(-1)  # Flatten to 1D tensor of size sequence_length * channel_dim
        sorted_values, sorted_indices = torch.sort(flattened, descending=True)

        # Top1
        top1_value = sorted_values[0].item()
        top1_values.append(top1_value)


        # Top3
        top3_values_i = sorted_values[:3].cpu().numpy()
        top3_indices_flat = sorted_indices[:3].cpu().numpy()
        # Map back to 2D indices
        print("top3_indices_flat",top3_indices_flat )
        print("channel_dim",channel_dim )

        
        top3_rows = top3_indices_flat // channel_dim
        top3_cols = top3_indices_flat % channel_dim
        top3_indices_i = list(zip(top3_rows, top3_cols))
        top3_values.append(top3_values_i)
        top3_indices.append(top3_indices_i)

        # Top1% elements
        top1_percent_count = max(1, int(np.ceil(num_elements * 0.01)))
        top1_percent = sorted_values[:top1_percent_count].cpu().numpy()
        top1_percent_values.append(top1_percent)

        # Median value
        median = torch.median(flattened).item()
        median_values.append(median)

    # Compute mean and standard deviation for each quantity
    top1_mean = np.mean(top1_values)
    top1_std = np.std(top1_values)

    top3_means = [np.mean(vals) for vals in top3_values]
    top3_mean = np.mean(top3_means)
    top3_std = np.std(top3_means)

    top1_percent_mean = np.mean([np.mean(vals) for vals in top1_percent_values])
    top1_percent_std = np.std([np.mean(vals) for vals in top1_percent_values])

    median_mean = np.mean(median_values)
    median_std = np.std(median_values)
    epoch = iteration//BATCH128

    # 통계 데이터를 probe_result.csv에 기록
    with open('/home/shkim/QT_DeiT_small/probe_result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # 파일이 비어 있으면 헤더 추가
            writer.writerow(['Epoch', 'Layer', 'Top1 Mean', 'Top1 Std', 'Top3 Mean', 'Top3 Std', 
                             'Top1% Mean', 'Top1% Std', 'Median Mean', 'Median Std'])
        writer.writerow([epoch, layer, top1_mean, top1_std, top3_mean, top3_std, 
                         top1_percent_mean, top1_percent_std, median_mean, median_std])

    # 각 배치의 Top3 인덱스를 top3_indices.csv에 기록
    with open('/home/shkim/QT_DeiT_small/top3_indices.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # 파일이 비어 있으면 헤더 추가
            writer.writerow(['Epoch', 'Layer', 'Sample Index', 'Rank', 'Row Index', 'Channel Index'])
        
        for i, indices in enumerate(top3_indices):
            for rank, (row, col) in enumerate(indices, 1):
                writer.writerow([epoch, layer, i, rank, row, col])