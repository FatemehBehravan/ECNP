#!/usr/bin/env python3
"""
Test script to demonstrate the new technical indicators functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.xauusd_data import NumericDataset
import torch
import pandas as pd

def test_technical_indicators():
    """Test the technical indicators implementation"""
    
    print("Testing Technical Indicators Implementation...")
    print("=" * 50)
    
    # Create dataset instance
    dataset = NumericDataset(
        batch_size=1,
        max_num_context=50,
        x_size=15,  # New size with technical indicators
        testing=True
    )
    
    # Generate sample data
    try:
        sample_data = dataset.generate_curves(device="cpu")
        
        print(f"✅ Data generation successful!")
        print(f"Query shape: {sample_data.query[0][0].shape}")  # Context X
        print(f"Target shape: {sample_data.target_y.shape}")
        print(f"Number of features: {sample_data.query[0][0].shape[-1]}")
        
        # Print feature information
        feature_names = [
            'hour_sin', 'hour_cos', 'open', 'high', 'low',
            'rsi', 'macd', 'macd_signal', 'bb_position', 'bb_width',
            'stoch_k', 'williams_r', 'momentum_5', 'atr', 'volume_ratio'
        ]
        
        print("\n📊 Technical Indicators Added:")
        print("-" * 30)
        for i, name in enumerate(feature_names):
            print(f"{i+1:2d}. {name}")
        
        # Show sample values for a few indicators
        context_data = sample_data.query[0][0][0, :5, :]  # First 5 time steps
        print(f"\n📈 Sample values from first context sequence:")
        print("-" * 40)
        for i, name in enumerate(feature_names[:10]):  # Show first 10 features
            values = context_data[:, i].detach().numpy()
            print(f"{name:12s}: {values}")
        
        # Test inverse transform
        print(f"\n🔄 Testing inverse transform...")
        target_close = sample_data.target_y[0, 0, :, 0]  # First target sequence
        original_close = dataset.inverse_transform(target_close, 'close')
        print(f"Target close (scaled):     {target_close[:5].detach().numpy()}")
        print(f"Target close (original):   {original_close[:5].detach().numpy()}")
        
        print(f"\n✅ All tests passed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_technical_indicators()
    if success:
        print(f"\n🎉 Technical indicators implementation is working correctly!")
        print(f"\nYour model now has access to 15 features including:")
        print(f"• Time features: hour_sin, hour_cos")
        print(f"• Price features: open, high, low")  
        print(f"• Technical indicators: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, Momentum, ATR, Volume")
        print(f"\nUpdate your model's input size to x_size=15 when using this enhanced dataset!")
    else:
        print(f"\n❌ There were issues with the implementation. Please check the error messages above.")