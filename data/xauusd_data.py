import pandas as pd
import collections
import os
import torch

# Define CustomRegressionDescription similar to NPRegressionDescription
CustomRegressionDescription = collections.namedtuple(
    "CustomRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points", "task_defn")
)

class DatasetReader(object):
    def __init__(self, file_path, max_num_context, device="cpu", split_ratio=(0.7, 0.15, 0.15)):
        """
        DatasetReader initializes with the CSV file and splits data into train, val, and test.
        :param file_path: Path to the CSV file.
        :param max_num_context: Maximum number of context points for training.
        :param device: Device to use (cpu or cuda).
        :param split_ratio: Tuple indicating train, validation, test split ratio.
        """
        self.file_path = file_path
        self.max_num_context = max_num_context
        self.device = device
        self.split_ratio = split_ratio

        # Load data from the CSV
        self.data = pd.read_csv(file_path)
        self.X = self.data.drop(columns=['close', 'low', 'high']).values  # Features
        self.y = self.data[['close', 'low', 'high']].values               # Labels
        
        # Split data
        self._split_data()
        
    def _split_data(self):
        """
        Splits the data into training, validation, and test sets based on the provided ratio.
        """
        total_samples = self.X.shape[0]
        train_size = int(self.split_ratio[0] * total_samples)
        val_size = int(self.split_ratio[1] * total_samples)

        self.X_train = self.X[:train_size]
        self.y_train = self.y[:train_size]

        self.X_val = self.X[train_size:train_size + val_size]
        self.y_val = self.y[train_size:train_size + val_size]

        self.X_test = self.X[train_size + val_size:]
        self.y_test = self.y[train_size + val_size:]
        
    def generate_curves(self, fixed_num_context=-1):
        """
        Generate the dataset in the format of CustomRegressionDescription.
        :param fixed_num_context: If provided, use fixed number of context points.
        :return: A CustomRegressionDescription namedTuple.
        """
        if fixed_num_context > 0:
            num_context = fixed_num_context
        else:
            num_context = torch.randint(low=3, high=self.max_num_context + 1, size=(1,)).item()
        
        # Use training data for context
        context_x = torch.Tensor(self.X_train[:num_context]).to(self.device)
        context_y = torch.Tensor(self.y_train[:num_context]).to(self.device)
        
        # Use validation data for target_x (query) and test data for target_y
        target_x = torch.Tensor(self.X_val).to(self.device)
        target_y = torch.Tensor(self.y_test).to(self.device)

        # For simplicity, setting task property as None (you can customize it)
        task_property = None
        
        # Prepare query as a tuple of (context_x, context_y) and target_x
        query = ((context_x, context_y), target_x)
        
        return CustomRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[0],
            num_context_points=num_context,
            task_defn=task_property
        )


# Example Usage
file_path = '/content/XAUUSD15_Data.csv'  # Update with the path to your CSV file

# Initialize DatasetReader with CSV data
dataset_reader = DatasetReader(file_path=file_path, max_num_context=50, device="cpu")

# Generate data in the format of CustomRegressionDescription
data = dataset_reader.generate_curves()

# Print the result
print(data)
