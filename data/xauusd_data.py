import torch
import collections
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points", "task_defn", "scaler", "datetime_data")
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
        self.scaler = None
        self.datetime_data = None

    def generate_curves(self, device, fixed_num_context=3, forecast_horizon=11):
        def load_csv_data(file_path):
            # Read only the last 500 rows directly
            df = pd.read_csv(file_path, nrows=501)
            
            # Vectorized operations instead of creating intermediate columns
            hours = pd.to_datetime(df['time'], unit='s').dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            
            # Only scale the columns we need
            columns_to_scale = ['hour_sin', 'open', 'high', 'low', 'close']
            self.scaler = MinMaxScaler()
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            
            # Return both scaled data and datetime
            return df[columns_to_scale], df['datetime']

        def create_xy_matrices(data, datetime_series, pre_length=10, post_length=10):
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

<<<<<<< HEAD
        file_path = './datasets/Test_XAUUSD.csv' if self._testing else './datasets/XAUUSD.csv'
=======
        # file_path = './datasets/Test_XAUUSD.csv' if self._testing else './datasets/XAUUSD.csv'
        file_path = './datasets/XAUUSD.csv'
>>>>>>> 654248f445830de39aa085d02d15a667473fa194
        df_scaled, df_datetime = load_csv_data(file_path)
        x_list, y_list = create_xy_matrices(df_scaled, df_datetime, pre_length=10, post_length=10)

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
            # Get datetime values corresponding to the data points
            datetime_values = df_datetime.values
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
            # Get datetime values for selected indices
            datetime_values = df_datetime.iloc[indices].values

        # Store datetime data for later use
        self.datetime_data = datetime_values

        # reshape to batch format
        x_values = x_values.unsqueeze(0).repeat(1, 1, 1, 1)  # [B, T, 10, 4]
        y_values = y_values.unsqueeze(0).repeat(1, 1, 1, 1)  # [B, T, 10, 1]

        # print(x_values.shape) #torch.Size([1, 72, 10, 4])
        # print(y_values.shape) #torch.Size([1, 72, 10, 1])


        task_property = torch.tensor(1).to(device)


        if self._testing:
            target_x = x_values
            target_y = y_values
            target_datetime = datetime_values

            idx = np.random.permutation(num_85_percent)[:num_context_points]
            idx = np.sort(idx)
            context_x = x_values[:, idx[:num_context_points], :, :]
            context_y = y_values[:, idx[:num_context_points], :, :]
            context_datetime = df_datetime.iloc[idx[:num_context_points]].values
        else: 
            target_x = x_values[:, :num_total_points, :, :]
            target_y = y_values[:, :num_total_points, :, :]
            target_datetime = datetime_values[:num_total_points]
            context_x = x_values[:, :num_context, :, :]
            context_y = y_values[:, :num_context, :, :]
            context_datetime = datetime_values[:num_context]

        query = ((context_x.to(device), context_y.to(device)), target_x.to(device))

        # Prepare datetime data for return
        datetime_info = {
            'target_datetime': target_datetime,
            'context_datetime': context_datetime
        }

        return NPRegressionDescription(
            query=query,
            target_y=target_y.to(device),
            num_total_points=num_total_points,
            num_context_points=num_context,
            task_defn=task_property,
            scaler=self.scaler,
            datetime_data=datetime_info
        )

    def inverse_transform(self, data, feature_name):
        """
        Transform scaled data back to original scale
        :param data: tensor or numpy array of shape [..., 1]
        :param feature_name: one of ['hour_sin', 'open', 'high', 'low', 'close']
        :return: data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Generate curves first.")
            
        feature_idx = {'hour_sin': 0, 'open': 1, 'high': 2, 'low': 3, 'close': 4}
        
        # Convert tensor to numpy if needed
        is_tensor = torch.is_tensor(data)
        if is_tensor:
            data = data.cpu().numpy()
            
        # Reshape to 2D
        original_shape = data.shape
        data_2d = data.reshape(-1, 1)
        
        # Create dummy array for inverse transform
        dummy = np.zeros((data_2d.shape[0], 5))  # 5 features
        dummy[:, feature_idx[feature_name]] = data_2d.ravel()
        
        # Inverse transform
        dummy = self.scaler.inverse_transform(dummy)
        result = dummy[:, feature_idx[feature_name]].reshape(original_shape)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            result = torch.from_numpy(result)
            
        return result
