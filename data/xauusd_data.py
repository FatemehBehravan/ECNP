import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class NumericDataset(Dataset):
    def __init__(self, file_path, features, labels, transform=None):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
            features (list): List of feature column names.
            labels (list): List of label column names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
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