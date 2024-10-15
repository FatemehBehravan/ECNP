import torch
import collections
import numpy as np
import random
import os
import pandas as pd

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points","task_defn")
)



class NumericDataset(object):


    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 testing=False,
                 device="cpu"):

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self._device=device



    def generate_curves(self, device, fixed_num_context=-1):
        file_path_train = './datasets/XAUUSD/train/train_data.csv'
        file_path_test = './datasets/XAUUSD/test/test_data.csv'

        def load_csv_data(file_path):
            data = pd.read_csv(file_path)
            data.dropna(subset=['date', 'open', 'close'], inplace=True)
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data['date'] = data['date'].astype(int) / 10**9
            data['open'] = data['open'].astype(int)
            data['colse'] = data['close'].astype(int)
            X = data['date'].values
            y = data['close'].values
            return X, y

        
        base_path = 'datasets/XAUUSD/'
        X_train, y_train = load_csv_data(os.path.join(base_path, 'train/train_data.csv'))
        X_val, y_val = load_csv_data(os.path.join(base_path, 'val/val_data.csv'))
        X_test, y_test = load_csv_data(os.path.join(base_path, 'test/test_data.csv'))


        feature_columns = [ 'date']
        label_columns = ['close']


        if fixed_num_context>0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))


        if self._testing:
            num_total_points = 400
            num_target = num_total_points
            x_values = torch.tensor(X_test[:self._batch_size * num_total_points], dtype=torch.float32)
            y_values = torch.tensor(y_test[:self._batch_size * num_total_points], dtype=torch.float32)
            x_values = x_values.view(self._batch_size, num_total_points, self._x_size)  # x_size=1
            y_values = y_values.view(self._batch_size, num_total_points, self._y_size)  # y_size=1

            # print(x_values.shape)
            # print(y_values.shape)

        else:
            num_target = torch.randint(3,self._max_num_context+1, size = (1,))
            num_total_points = num_target + num_context
            x_values = torch.tensor(X_train[:self._batch_size * num_total_points], dtype=torch.float32)
            y_values = torch.tensor(y_train[:self._batch_size * num_total_points], dtype=torch.float32)
            x_values = x_values.view(self._batch_size, num_total_points, self._x_size)  # x_size=1
            y_values = y_values.view(self._batch_size, num_total_points, self._y_size)  # y_size=1
            # print(x_values.shape)
            # print(y_values.shape)

        task_property = torch.tensor(1)
        if self._testing:
            target_x = x_values
            target_y = y_values
            # print('here',x_values.shape)
            # print(y_values.shape)
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:,idx[:num_context],:]
        else:
            target_x = x_values[:,:num_target+num_context,:]
            target_y = y_values[:,:num_target+num_context,:]

            context_x = x_values[:,:num_context,:]
            context_y = y_values[:,:num_context,:]


        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        task_property = task_property.to(device)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
            task_defn=task_property
        )
