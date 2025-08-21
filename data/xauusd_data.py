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

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for trading data
    """
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = df['close'].ewm(span=5, min_periods=1).mean()
    df['ema_10'] = df['close'].ewm(span=10, min_periods=1).mean()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    df['rsi'] = calculate_rsi(df['close'])
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['close'].ewm(span=12, min_periods=1).mean()
    ema_26 = df['close'].ewm(span=26, min_periods=1).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_window = 20
    bb_std = 2
    bb_sma = df['close'].rolling(window=bb_window, min_periods=1).mean()
    bb_std_dev = df['close'].rolling(window=bb_window, min_periods=1).std()
    df['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
    df['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
    df['bb_middle'] = bb_sma
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, window=14):
        lowest_low = low.rolling(window=window, min_periods=1).min()
        highest_high = high.rolling(window=window, min_periods=1).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()
        return stoch_k.fillna(50), stoch_d.fillna(50)
    
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    
    # Williams %R
    def calculate_williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r.fillna(-50)
    
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])
    
    # Momentum indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['price_rate_change'] = df['close'].pct_change(periods=10)
    
    # Average True Range (ATR)
    def calculate_atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        return atr
    
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Volume indicators (if volume data is available)
    if 'tick_volume' in df.columns and df['tick_volume'].sum() > 0:
        df['volume_sma'] = df['tick_volume'].rolling(window=10, min_periods=1).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['tick_volume']
    else:
        df['volume_sma'] = 0
        df['volume_ratio'] = 1
        df['price_volume'] = df['close']
    
    # Fill any remaining NaN values
    df = df.ffill().bfill()
    
    return df

class NumericDataset(object):
    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=10,  # Updated to include 12 technical indicators
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

    def generate_curves(self, device, fixed_num_context=3, forecast_horizon=21):
        def load_csv_data(file_path):
            # Read only the last 500 rows directly
            df = pd.read_csv(file_path, nrows=501)
            
            # Add time-based features
            hours = pd.to_datetime(df['time'], unit='s').dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Define feature columns to include in the model
            feature_columns = [
                'hour_sin', 'open', 'high', 'low',
                'rsi', 'macd', 'macd_signal', 'bb_position',
                'stoch_k', 'momentum_5'
            ]
            
            # Target column remains the same
            target_columns = ['close']
            
            # Combine all columns for scaling
            columns_to_scale = feature_columns + target_columns
            
            # Scale the data
            self.scaler = MinMaxScaler()
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            
            # Return both scaled data and datetime
            return df[columns_to_scale], df['datetime']

        def create_xy_matrices(data, datetime_series, pre_length, post_length):
            feature_cols = [
                'hour_sin', 'open', 'high', 'low',
                'rsi', 'macd', 'macd_signal', 'bb_position',
                'stoch_k', 'momentum_5'
            ]
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
        # file_path = './datasets/Test_XAUUSD.csv' if self._testing else './datasets/XAUUSD.csv'
        file_path = './datasets/XAUUSD.csv'

        df_scaled, df_datetime = load_csv_data(file_path)
        x_list, y_list = create_xy_matrices(df_scaled, df_datetime, pre_length=20, post_length=20)

        num_points = len(x_list)
        num_85_percent = int(num_points * 0.85)

        # تبدیل به tensor
        x_all = torch.tensor(np.array(x_list), dtype=torch.float32)  # [N, 10, 12]
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
        x_values = x_values.unsqueeze(0).repeat(1, 1, 1, 1)  # [B, T, 10, 12]
        y_values = y_values.unsqueeze(0).repeat(1, 1, 1, 1)  # [B, T, 10, 1]

        # print(x_values.shape) #torch.Size([1, 72, 10, 12])
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
        :param feature_name: one of the available features
        :return: data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Generate curves first.")
            
        feature_idx = {
            'hour_sin': 0, 'open': 1, 'high': 2, 'low': 3, 'rsi': 4, 
            'macd': 5, 'macd_signal': 6, 'bb_position': 7,
            'stoch_k': 8, 'momentum_5': 9, 
            'close': 10
        }
        
        # Convert tensor to numpy if needed
        is_tensor = torch.is_tensor(data)
        if is_tensor:
            data = data.cpu().numpy()
            
        # Reshape to 2D
        original_shape = data.shape
        data_2d = data.reshape(-1, 1)
        
        # Create dummy array for inverse transform
        dummy = np.zeros((data_2d.shape[0], 11))  # 13 features (12 features + 1 target)
        dummy[:, feature_idx[feature_name]] = data_2d.ravel()
        
        # Inverse transform
        dummy = self.scaler.inverse_transform(dummy)
        result = dummy[:, feature_idx[feature_name]].reshape(original_shape)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            result = torch.from_numpy(result)
            
        return result

