import torch
import collections
import numpy as np
import random
import os
import pandas as pd

NPRegressionDescription = collections.namedtuple("NPRegressionDescription",("query", "target_y", "num_total_points", "num_context_points","task_defn"))



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




    def generate_curves(self, device, fixed_num_context=-1):


        def load_csv_data(file_path):
          
            data = pd.read_csv(file_path)
            data['counter'] = data['counter'].astype(int)
            #data['open'] = data['open'].astype(int)
            data['colse'] = data['close'].astype(int)
            data['MA'] = data['close'].rolling(window=100).mean()
            data = data.tail(400)

            x = data['counter'].values
            y = data['MA'].values
            return x, y


        if fixed_num_context>0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self._max_num_context + 1, size=(1,))

        file_path = './datasets/XAUUSD.csv'
        x_values, y_values = load_csv_data(os.path.join(file_path))

        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = x_values[-num_target:]
            y_values = y_values[-num_target:]

        else:
            num_target = torch.randint(3,self._max_num_context+1, size = (1,))
            num_total_points = num_target + num_context
            shuffled_indices = np.random.permutation(len(x_values))
            random_indices = shuffled_indices[:num_total_points]

            x_values = x_values[random_indices]
            y_values = y_values[random_indices]



        x_values = torch.tensor(x_values, dtype=torch.float32)
        y_values = torch.tensor(y_values, dtype=torch.float32)
        x_values = x_values.view(self._x_size, num_total_points, self._x_size)  # x_size=1
        y_values = y_values.view(self._y_size, num_total_points, self._y_size)  # y_size=1


        task_property = torch.tensor(1)

        if self._testing:
            print('testing_step')
            target_x = x_values
            target_y = y_values
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:,idx[:num_context],:]
        else:
            print('training_step')
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