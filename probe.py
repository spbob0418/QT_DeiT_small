import torch
import numpy as np
import csv
import os

BATCH256 = 1252
BATCH128 = 2502

def process_tensor(x, iteration, layer: str, block_num):
    # x: Tensor of shape [BS, sequence_length, channel_dim]
    with torch.no_grad():
        if x.dim() == 2: 
            x = x.unsqueeze(0) 
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

    # probe_result.csv 파일 경로에 layer 이름 추가
    probe_result_path = f'/home/shkim/QT_DeiT_small/probe_report/probe_result_{layer}.csv'
    top3_indices_path = f'/home/shkim/QT_DeiT_small/probe_report/top3_indices_{layer}.csv'

    # 통계 데이터를 각 layer에 맞는 probe_result 파일에 기록
    with open(probe_result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(probe_result_path) == 0:  # 파일이 비어 있으면 헤더 추가
            writer.writerow(['Epoch', 'Block', 'Layer', 'Top1 Mean', 'Top1 Std', 'Top3 Mean', 'Top3 Std', 
                            'Top1% Mean', 'Top1% Std', 'Median Mean', 'Median Std'])
        # 소수 둘째 자리까지만 기록
        writer.writerow([epoch, block_num, layer,
                        '{:.2f}'.format(top1_mean), '{:.2f}'.format(top1_std),
                        '{:.2f}'.format(top3_mean), '{:.2f}'.format(top3_std),
                        '{:.2f}'.format(top1_percent_mean), '{:.2f}'.format(top1_percent_std),
                        '{:.2f}'.format(median_mean), '{:.2f}'.format(median_std)])

    # 각 배치의 Top3 인덱스를 layer별 top3_indices 파일에 기록
    with open(top3_indices_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(top3_indices_path) == 0:  # 파일이 비어 있으면 헤더 추가
            writer.writerow(['Epoch', 'Block', 'Layer', 'Sample Index', 'Rank', 'Row Index', 'Channel Index'])
        
        for i, indices in enumerate(top3_indices):
            for rank, (row, col) in enumerate(indices, 1):
                writer.writerow([epoch, block_num, layer, i, rank, row, col])