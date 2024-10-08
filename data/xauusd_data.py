import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import collections

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points","task_defn")
)

class NumericDataset(Dataset):
    def __init__(self, file_path, features, labels, transform=None):
        self.data = pd.read_csv(file_path)
        for col in self.data.columns:
            if 'date' in col.lower():
                self.data[col] = pd.to_datetime(self.data[col]).astype(int) / 10**9 

        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Extract features and labels
        X = sample[self.features].values.astype('float32')
        y = sample[self.labels].values.astype('float32')

        if self.transform:
            X = self.transform(X)

        return torch.tensor(X), torch.tensor(y)

    def get_context_target(self, device, fixed_num_context=-1):

        if fixed_num_context > 0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self.max_num_context + 1, size=(1,)).item()

        X = self.data[self.features].values
        y = self.data[self.labels].values
        num_total_points = X.shape[0]

        if self.testing:
            num_target = num_total_points
        else:
            num_target = torch.randint(3, self.max_num_context + 1, size=(1,)).item()

        idx = torch.randperm(num_total_points)
        context_idx = idx[:num_context]
        target_idx = idx[:num_target + num_context]

        context_x = torch.tensor(X[context_idx], dtype=torch.float32).to(device)
        context_y = torch.tensor(y[context_idx], dtype=torch.float32).to(device)

        target_x = torch.tensor(X[target_idx], dtype=torch.float32).to(device)
        target_y = torch.tensor(y[target_idx], dtype=torch.float32).to(device)

        query = ((context_x, context_y), target_x)

        task_property = torch.Tensor([self.max_num_context, num_total_points]).to(device)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=num_total_points,
            num_context_points=num_context,
            task_defn=task_property
        )
