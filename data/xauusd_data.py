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
                 x_size=4,  # برای ['hour_sin', 'open', 'high', 'low']
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
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['time'], unit='s')
            df['hour'] = df['date'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['minute'] = df['date'].dt.minute
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df = df.tail(400)

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['hour_sin', 'open', 'high', 'low', 'close']])
            scaled_data = pd.DataFrame(scaled_data, columns=['hour_sin', 'open', 'high', 'low', 'close'])
            x = scaled_data[['hour_sin', 'open', 'high', 'low']].values
            y = scaled_data['close'].values
            return scaled_data, x, y

        # بارگذاری داده‌ها
        file_path = './datasets/XAUUSD.csv'
        df, x_values, y_values = load_csv_data(file_path)
        num_points = len(x_values)
        num_85_percent = int(num_points * 0.85)

        # تعداد نقاط (مثلاً 3)
        num_context = fixed_num_context if fixed_num_context > 0 else torch.randint(low=3, high=self._max_num_context + 1, size=(1,)).item()

        # تنظیم context_length و target_length
        context_length = 21  # 20 روز گذشته + روز فعلی
        target_length = forecast_horizon  # روز فعلی + 10 روز آینده

        if self._testing:
            num_target = num_points
            num_total_points = num_target
            start_indices = list(range(20, num_points - target_length + 1))  # همه نقاط ممکن به ترتیب زمانی
            num_context = len(start_indices)  # تعداد نقاط زمینه برابر با تعداد نقاط ممکن
        else:
            num_target = num_context  # برای سادگی، num_target را برابر با num_context قرار می‌دهیم
            num_total_points = num_target
            # انتخاب نقاط شروع به ترتیب زمانی
            start_indices = []
            for i in range(num_context):
                # باید حداقل 20 روز قبل و 10 روز بعد داشته باشیم
                start_idx = torch.randint(20, num_85_percent - target_length, (1,)).item()
                start_indices.append(start_idx)
            start_indices.sort()  # ترتیب زمانی

        # تولید context_x, context_y, target_x, target_y برای هر نقطه
        context_x_list = []
        context_y_list = []
        target_x_list = []
        target_y_list = []

        for start_idx in start_indices:
            # Context_x: 20 روز گذشته + روز فعلی (ماتریس 21×4)
            context_start = max(0, start_idx - 20)
            context_end = start_idx + 1
            context_data = df.iloc[context_start:context_end][['hour_sin', 'open', 'high', 'low']].values
            if context_data.shape[0] < 21:  # پر کردن برای نقاط اولیه
                padding_size = 21 - context_data.shape[0]
                padding = np.mean(context_data, axis=0, keepdims=True).repeat(padding_size, axis=0)
                context_data = np.vstack((padding, context_data))
            context_x = torch.tensor(context_data, dtype=torch.float32).view(1, 1, 21, 4)  # (1, 1, 21, 4)

            # Context_y: 20 روز گذشته + روز فعلی (ماتریس 21×1)
            context_y_data = df.iloc[context_start:context_end]['close'].values
            if len(context_y_data) < 21:
                padding_size = 21 - len(context_y_data)
                padding = np.mean(context_y_data).repeat(padding_size)
                context_y_data = np.concatenate((context_y_data, padding))
            context_y = torch.tensor(context_y_data, dtype=torch.float32).view(1, 1, 21, 1)  # (1, 1, 21, 1)

            # Target_x: روز فعلی + 10 روز آینده (ماتریس 11×4)
            target_start = start_idx
            target_end = start_idx + target_length
            target_data = df.iloc[target_start:target_end][['hour_sin', 'open', 'high', 'low']].values
            if len(target_data) < target_length:  # پر کردن برای نقاط انتهایی
                padding_size = target_length - len(target_data)
                padding = np.mean(target_data, axis=0, keepdims=True).repeat(padding_size, axis=0)
                target_data = np.vstack((padding, target_data))
            target_x = torch.tensor(target_data, dtype=torch.float32).view(1, 1, 11, 4)  # (1, 1, 11, 4)

            # Target_y: روز فعلی + 10 روز آینده (ماتریس 11×1)
            target_y_data = df.iloc[target_start:target_end]['close'].values
            if len(target_y_data) < target_length:
                padding_size = target_length - len(target_y_data)
                padding = np.mean(target_y_data).repeat(padding_size)
                target_y_data = np.concatenate((target_y_data, padding))
            target_y = torch.tensor(target_y_data, dtype=torch.float32).view(1, 1, 11, 1)  # (1, 1, 11, 1)

            context_x_list.append(context_x)
            context_y_list.append(context_y)
            target_x_list.append(target_x)
            target_y_list.append(target_y)

        # ترکیب تمام نقاط
        context_x = torch.cat(context_x_list, dim=1).to(device)  # (1, num_context, 21, 4)
        context_y = torch.cat(context_y_list, dim=1).to(device)  # (1, num_context, 21, 1)
        target_x = torch.cat(target_x_list, dim=1).to(device)  # (1, num_target, 11, 4)
        target_y = torch.cat(target_y_list, dim=1).to(device)  # (1, num_target, 11, 1)

        task_property = torch.tensor(1).to(device)

        query = ((context_x, context_y), target_x)
        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],  # num_target
            num_context_points=context_x.shape[1],  # num_context
            task_defn=task_property
        )