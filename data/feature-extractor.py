import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator

class FeatureExtractor:
    """
    A class to extract features (ATR and SMA) for each data point in an XAUUSD dataset.
    """

    def __init__(self, atr_window=14, sma_window=20):
        """
        Initialize the FeatureExtractor with parameters for ATR and SMA.

        :param atr_window: The window size for ATR calculation (default: 14).
        :param sma_window: The window size for SMA calculation (default: 20).
        """
        self.atr_window = atr_window
        self.sma_window = sma_window

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:

        def load_csv_data(file_path):
            data = pd.read_csv(file_path)
            data.dropna(subset=['date', 'open', 'close', 'high', 'low'], inplace=True)
            
            data['colse'] = data['open'].astype(int)
            data['colse'] = data['high'].astype(int)
            data['colse'] = data['low'].astype(int)
            data['colse'] = data['close'].astype(int)

            columns = data[['open', 'high', 'low', 'close']].values
            return columns


        base_path = 'datasets/XAUUSD/'
        load_csv_data(os.path.join(base_path, 'train/train_data.csv'))
        load_csv_data(os.path.join(base_path, 'test/test_data.csv'))

        # Validate input data
        required_columns = 
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data must contain the '{col}' column.")

        # Calculate ATR
        atr_indicator = AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.atr_window
        )
        data['ATR'] = atr_indicator.average_true_range()

        # Calculate SMA
        sma_indicator = SMAIndicator(
            close=data['close'],
            window=self.sma_window
        )
        data['SMA'] = sma_indicator.sma_indicator()

        # Handle NaN values (e.g., at the beginning of the dataset)
        data = data.dropna().reset_index(drop=True)

        return data

# Example usage
if __name__ == "__main__":
    # Create a sample dataset
    data = pd.DataFrame({
        'open': [1900, 1905, 1910, 1920, 1915],
        'high': [1910, 1915, 1920, 1930, 1925],
        'low': [1895, 1900, 1905, 1915, 1910],
        'close': [1905, 1910, 1915, 1925, 1920]
    })

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(atr_window=14, sma_window=3)

    # Extract features
    data_with_features = feature_extractor.extract_features(data)

    # Display the dataset with features
    print(data_with_features)
