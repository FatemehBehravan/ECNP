import torch
import collections
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points", "task_defn")
)

class NumericDataset(object):
    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=4,  # ['hour_sin', 'open', 'high', 'low']
                 y_size=1,
                 testing=False,
                 device="cpu"):
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self._device = device
        self.start_index = 0

    def generate_curves(self, device, fixed_num_context=3, forecast_horizon=11):
        def load_csv_data(file_path):
            # Read only the last 500 rows directly
            df = pd.read_csv(file_path, nrows=501)
            
            # Vectorized operations instead of creating intermediate columns
            hours = pd.to_datetime(df['time'], unit='s').dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            
            # Only scale the columns we need
            columns_to_scale = ['hour_sin', 'open', 'high', 'low', 'close']
            scaler = MinMaxScaler()
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            
            return df[columns_to_scale]

        def create_xy_matrices(data, pre_length=10, post_length=10):
            feature_cols = ['hour_sin', 'open', 'high', 'low']
            target_col = 'close'
            
            # Pre-allocate arrays for better performance
            num_samples = len(data) - pre_length - post_length
            x_list = np.zeros((num_samples, pre_length, len(feature_cols)))
            y_list = np.zeros((num_samples, post_length, 1))
            
            # Use numpy operations instead of list appending
            for i in range(num_samples):
                x_list[i] = data.iloc[i:i + pre_length][feature_cols].to_numpy()#0-10
                y_list[i] = data.iloc[i + pre_length + 1 :i + pre_length + post_length + 1][[target_col]].to_numpy() #11-21
            
            return x_list, y_list

        file_path = './datasets/Test_XAUUSD.csv' if self._testing else './datasets/XAUUSD.csv'
        df_scaled = load_csv_data(file_path)
        x_list, y_list = create_xy_matrices(df_scaled, pre_length=10, post_length=10)

        num_points = len(x_list)
        num_85_percent = int(num_points * 0.85)

        # تبدیل به tensor
        x_all = torch.tensor(np.array(x_list), dtype=torch.float32)  # [N, 10, 4]
        y_all = torch.tensor(np.array(y_list), dtype=torch.float32)  # [N, 10, 1]


        # تعیین تعداد context و target points
        num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,)).item()

        if self._testing:
            x_values = x_all
            y_values = y_all
            num_target = x_values.shape[0]
            num_total_points = num_target
            num_context_points = min(self._max_num_context, num_85_percent)
        else:
            num_target = torch.randint(3, self._max_num_context + 1, size=(1,)).item()
            num_total_points = num_target + num_context

            indices = np.random.permutation(num_85_percent)[:num_total_points]
            indices = np.sort(indices)
            x_values = x_all[indices]
            y_values = y_all[indices]

        # reshape to batch format
        x_values = x_values.unsqueeze(0).repeat(1, 1, 1, 1)  # [B, T, 10, 4]
        y_values = y_values.unsqueeze(0).repeat(1, 1, 1, 1)  # [B, T, 10, 1]

        # print(x_values.shape) #torch.Size([1, 72, 10, 4])
        # print(y_values.shape) #torch.Size([1, 72, 10, 1])


        task_property = torch.tensor(1).to(device)


        if self._testing:
            target_x = x_values
            target_y = y_values

            idx = np.random.permutation(num_85_percent)[:num_context_points]
            idx = np.sort(idx)
            context_x = x_values[:, idx[:num_context_points], :, :]
            context_y = y_values[:, idx[:num_context_points], :, :]
        else: 
            target_x = x_values[:, :num_total_points, :, :]
            target_y = y_values[:, :num_total_points, :, :]
            context_x = x_values[:, :num_context, :, :]
            context_y = y_values[:, :num_context, :, :]

        query = ((context_x.to(device), context_y.to(device)), target_x.to(device))


        return NPRegressionDescription(
            query=query,
            target_y=target_y.to(device),
            num_total_points=num_total_points,
            num_context_points=num_context,
            task_defn=task_property
        )
