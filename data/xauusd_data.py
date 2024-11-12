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
                 device="cpu",
                 ):

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self._device=device
        self.start_index = 0
        



    def generate_curves(self, device, fixed_num_context):
        file_path_train = './datasets/XAUUSD/train/train_data.csv'
        file_path_test = './datasets/XAUUSD/test/test_data.csv'

        def load_csv_data(file_path):
            data = pd.read_csv(file_path)
            data.dropna(subset=['date', 'open', 'close'], inplace=True)
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            # data = data[data.date > '2024-01-01']
            base_date = data['date'].min()
            data['date'] = (data['date'] - base_date).dt.total_seconds()
            data['date'] = data['date'].astype(int)
            #data['open'] = data['open'].astype(int)
            data['colse'] = data['close'].astype(int)
            
            x = data['date'].values
            y = data['close'].values
            return x, y

        
        base_path = 'datasets/XAUUSD/'
        x_train, y_train = load_csv_data(os.path.join(base_path, 'train/train_data.csv'))
        x_val, y_val = load_csv_data(os.path.join(base_path, 'val/val_data.csv'))
        x_test, y_test = load_csv_data(os.path.join(base_path, 'test/test_data.csv'))


        # feature_columns = [ 'date']
        # label_columns = ['close']

        if fixed_num_context>0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))

        window_size = 96
        step = 16
        
        if self._testing:
            num_total_points = 20
            num_target = num_total_points
            
            # print(self.start_index)
            if self.start_index + window_size > len(x_test):
                print("End of data reached")
                self.start_index = 0
            else:
                
                window_data_x = x_test[self.start_index:self.start_index + window_size]
                window_data_y = y_test[self.start_index:self.start_index + window_size]

                if len(window_data_x) < self._batch_size * num_total_points:
                    raise ValueError(f"Not enough data points: expected {self._batch_size * num_total_points}, got {len(window_data_x)}")

                
                x_values = torch.tensor(window_data_x[:self._batch_size * num_total_points], dtype=torch.float32)
                y_values = torch.tensor(window_data_y[:self._batch_size * num_total_points], dtype=torch.float32)

                x_values = x_values.view(self._batch_size, num_total_points, self._x_size) 
                y_values = y_values.view(self._batch_size, num_total_points, self._y_size)  
                self.start_index += step

                  


            # x_values = torch.tensor(x_test[:self._batch_size * num_total_points], dtype=torch.float32)
            # y_values = torch.tensor(y_test[:self._batch_size * num_total_points], dtype=torch.float32)
            # x_values = x_values.view(self._batch_size, num_total_points, self._x_size)  # x_size=1
            # y_values = y_values.view(self._batch_size, num_total_points, self._y_size)  # y_size=1

            # print(x_values.shape)
            # print(y_values.shape)

        else:
            num_target = 2
            num_total_points = num_target + num_context
            
            if self.start_index + window_size > len(x_train):
                print("End of data reached")
                self.start_index = 0
            else:
                # print('first_step_train')
                window_data_x = x_train[self.start_index:self.start_index + window_size]
                window_data_y = y_train[self.start_index:self.start_index + window_size]

                if len(window_data_x) < self._batch_size * num_total_points:
                    raise ValueError(f"Not enough data points: expected {self._batch_size * num_total_points}, got {len(window_data_x)}")

                
                x_values = torch.tensor(window_data_x[:self._batch_size * num_total_points], dtype=torch.float32)
                y_values = torch.tensor(window_data_y[:self._batch_size * num_total_points], dtype=torch.float32)

                x_values = x_values.view(self._batch_size, num_total_points, self._x_size) 
                y_values = y_values.view(self._batch_size, num_total_points, self._y_size)  
                self.start_index += step




            # x_values = torch.tensor(x_train[:self._batch_size * num_total_points], dtype=torch.float32)
            # y_values = torch.tensor(y_train[:self._batch_size * num_total_points], dtype=torch.float32)
            # x_values = x_values.view(self._batch_size, num_total_points, self._x_size)  # x_size=1
            # y_values = y_values.view(self._batch_size, num_total_points, self._y_size)  # y_size=1
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
            # print('secound_step_train')
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
