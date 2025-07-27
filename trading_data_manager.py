import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
import collections

# Import existing data components
from data.xauusd_data import NPRegressionDescription

class TradingDataManager:
    """
    Manages large datasets for continuous trading predictions
    Handles data streaming, preprocessing, and batch generation
    """
    
    def __init__(self, sequence_length=10, batch_size=1, device="cuda"):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        self.data_buffer = []
        self.current_index = 0
        self.is_fitted = False
        
    def load_extended_data(self, data_file_path, max_rows=None):
        """
        Load large dataset for trading simulation
        
        Args:
            data_file_path: Path to the CSV file with XAUUSD data
            max_rows: Maximum number of rows to load (None for all)
        """
        print(f"Loading trading data from {data_file_path}")
        
        if max_rows:
            df = pd.read_csv(data_file_path, nrows=max_rows)
        else:
            df = pd.read_csv(data_file_path)
            
        # Add time features
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        hours = df['datetime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        
        # Prepare feature columns
        self.feature_cols = ['hour_sin', 'open', 'high', 'low']
        self.target_col = 'close'
        self.datetime_col = 'datetime'
        
        # Store original prices for trading calculations
        self.original_prices = df['close'].values.copy()
        self.original_datetimes = df['datetime'].values.copy()
        
        # Scale the data
        columns_to_scale = ['hour_sin', 'open', 'high', 'low', 'close']
        scaled_data = self.scaler.fit_transform(df[columns_to_scale])
        
        # Create scaled dataframe
        self.scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
        self.scaled_df['datetime'] = df['datetime']
        
        print(f"Loaded {len(df)} data points")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        self.is_fitted = True
        return df
    
    def get_context_data(self, end_index, context_length=50):
        """
        Get context data for prediction at a specific time point
        
        Args:
            end_index: Current time index
            context_length: Number of historical points to use as context
            
        Returns:
            Context data in the format expected by the model
        """
        if not self.is_fitted:
            raise ValueError("Data not loaded. Call load_extended_data first.")
            
        start_idx = max(0, end_index - context_length)
        end_idx = end_index
        
        if end_idx - start_idx < self.sequence_length:
            return None
            
        # Extract context sequences
        context_sequences = []
        context_targets = []
        
        for i in range(start_idx, end_idx - self.sequence_length):
            # Input sequence (10 time steps of features)
            x_seq = self.scaled_df.iloc[i:i + self.sequence_length][self.feature_cols].values
            # Target sequence (10 time steps of close prices)
            y_seq = self.scaled_df.iloc[i + 1:i + self.sequence_length + 1][[self.target_col]].values
            
            context_sequences.append(x_seq)
            context_targets.append(y_seq)
        
        if not context_sequences:
            return None
            
        # Convert to tensors
        context_x = torch.tensor(np.array(context_sequences), dtype=torch.float32)
        context_y = torch.tensor(np.array(context_targets), dtype=torch.float32)
        
        # Add batch dimension and reshape for model
        context_x = context_x.unsqueeze(0)  # [1, N, 10, 4]
        context_y = context_y.unsqueeze(0)  # [1, N, 10, 1]
        
        return {
            'context_x': context_x.to(self.device),
            'context_y': context_y.to(self.device),
            'datetime': self.original_datetimes[end_index],
            'current_price': self.original_prices[end_index],
            'current_price_scaled': self.scaled_df.iloc[end_index]['close']
        }
    
    def get_target_data(self, start_index, target_length=20):
        """
        Get target data for prediction (future sequences to predict)
        
        Args:
            start_index: Starting index for target sequences
            target_length: Number of future sequences to predict
            
        Returns:
            Target data in model format
        """
        if start_index + target_length + self.sequence_length >= len(self.scaled_df):
            target_length = len(self.scaled_df) - start_index - self.sequence_length - 1
            
        if target_length <= 0:
            return None
            
        target_sequences = []
        target_targets = []
        
        for i in range(start_index, start_index + target_length):
            if i + self.sequence_length >= len(self.scaled_df):
                break
                
            # Input sequence for target prediction
            x_seq = self.scaled_df.iloc[i:i + self.sequence_length][self.feature_cols].values
            # Target sequence (what we want to predict)
            y_seq = self.scaled_df.iloc[i + 1:i + self.sequence_length + 1][[self.target_col]].values
            
            target_sequences.append(x_seq)
            target_targets.append(y_seq)
        
        if not target_sequences:
            return None
            
        # Convert to tensors
        target_x = torch.tensor(np.array(target_sequences), dtype=torch.float32)
        target_y = torch.tensor(np.array(target_targets), dtype=torch.float32)
        
        # Add batch dimension
        target_x = target_x.unsqueeze(0)  # [1, N, 10, 4]
        target_y = target_y.unsqueeze(0)  # [1, N, 10, 1]
        
        return {
            'target_x': target_x.to(self.device),
            'target_y': target_y.to(self.device)
        }
    
    def create_model_input(self, context_data, target_data):
        """
        Create input in the format expected by the forecasting model
        
        Args:
            context_data: Context data from get_context_data
            target_data: Target data from get_target_data
            
        Returns:
            NPRegressionDescription object for model input
        """
        if context_data is None or target_data is None:
            return None
            
        # Create query tuple as expected by the model
        query = ((context_data['context_x'], context_data['context_y']), 
                target_data['target_x'])
        
        # Create datetime information
        datetime_info = {
            'target_datetime': [context_data['datetime']],
            'context_datetime': [context_data['datetime']]
        }
        
        return NPRegressionDescription(
            query=query,
            target_y=target_data['target_y'],
            num_total_points=target_data['target_x'].shape[1],
            num_context_points=context_data['context_x'].shape[1],
            task_defn=torch.tensor(1).to(self.device),
            scaler=self.scaler,
            datetime_data=datetime_info
        )
    
    def inverse_transform_price(self, scaled_price):
        """
        Convert scaled price back to original scale
        
        Args:
            scaled_price: Price in scaled format
            
        Returns:
            Price in original scale
        """
        # Create dummy array with zeros for other features
        dummy = np.zeros((1, 5))  # [hour_sin, open, high, low, close]
        dummy[0, 4] = scaled_price  # close price is at index 4
        
        # Inverse transform
        original = self.scaler.inverse_transform(dummy)
        return original[0, 4]
    
    def get_price_at_index(self, index):
        """Get original price at specific index"""
        if 0 <= index < len(self.original_prices):
            return self.original_prices[index]
        return None
    
    def get_datetime_at_index(self, index):
        """Get datetime at specific index"""
        if 0 <= index < len(self.original_datetimes):
            return self.original_datetimes[index]
        return None
    
    def data_iterator(self, start_index=100, step_size=1, max_iterations=None):
        """
        Iterator for streaming data processing
        
        Args:
            start_index: Starting index for iteration
            step_size: Step size for moving through data
            max_iterations: Maximum number of iterations
            
        Yields:
            Data dictionaries for each time step
        """
        current_idx = start_index
        iteration_count = 0
        
        while current_idx < len(self.scaled_df) - 10:  # REDUCED: Smaller buffer for more trading opportunities
            if max_iterations and iteration_count >= max_iterations:
                break
                
            # Get context data
            context_data = self.get_context_data(current_idx, context_length=50)
            
            if context_data is not None:
                yield {
                    'index': current_idx,
                    'context_data': context_data,
                    'current_price': self.original_prices[current_idx],
                    'datetime': self.original_datetimes[current_idx]
                }
                
            current_idx += step_size
            iteration_count += 1
    
    def create_extended_dataset(self, output_file, num_samples=10000, base_data_file="datasets/XAUUSD.csv"):
        """
        Create an extended dataset by augmenting existing data
        (This is a utility function to create larger datasets if needed)
        
        Args:
            output_file: Path for the output extended dataset
            num_samples: Target number of samples
            base_data_file: Original data file to extend
        """
        print(f"Creating extended dataset with {num_samples} samples")
        
        # Load base data
        base_df = pd.read_csv(base_data_file)
        base_len = len(base_df)
        
        if num_samples <= base_len:
            print("Requested samples less than base data. Using original data.")
            base_df.to_csv(output_file, index=False)
            return
        
        # Create extended data by interpolation and noise addition
        extended_rows = []
        
        for i in range(num_samples):
            # Use cyclic indexing with small variations
            base_idx = i % base_len
            base_row = base_df.iloc[base_idx].copy()
            
            # Add small random variations to prices (±0.1%)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                noise_factor = np.random.normal(1.0, 0.001)  # 0.1% noise
                base_row[col] *= noise_factor
            
            # Adjust time (add small time increments)
            if i >= base_len:
                time_increment = (i - base_len) * 900  # 15 minutes increments
                base_row['time'] += time_increment
            
            extended_rows.append(base_row)
        
        extended_df = pd.DataFrame(extended_rows)
        extended_df = extended_df.reset_index(drop=True)
        extended_df['counter'] = range(len(extended_df))
        
        # Save extended dataset
        extended_df.to_csv(output_file, index=False)
        print(f"Extended dataset saved to {output_file}")
        print(f"Dataset size: {len(extended_df)} samples")
        
        return extended_df

def main():
    """Test the TradingDataManager"""
    print("Testing TradingDataManager")
    
    # Initialize data manager
    data_manager = TradingDataManager(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    df = data_manager.load_extended_data("datasets/Strategy_XAUUSD.csv")
    
    # Test data iteration
    print("\nTesting data iteration:")
    for i, data_point in enumerate(data_manager.data_iterator(start_index=100, max_iterations=5)):
        print(f"Step {i}: Index {data_point['index']}, Price: {data_point['current_price']:.2f}")
    
    print("\nTradingDataManager test completed successfully!")

if __name__ == "__main__":
    main() 