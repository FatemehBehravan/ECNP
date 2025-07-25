#!/usr/bin/env python3
"""
Simple script to run the XAUUSD trading strategy
This demonstrates how to use the trading system separately from the original forecasting code
"""

import os
import sys
from trading_strategy_complete import XAUUSDTradingStrategy

def run_strategy_demo():
    """
    Demonstrate the trading strategy with different configurations
    """
    print("=" * 80)
    print("XAUUSD TRADING STRATEGY DEMONSTRATION")
    print("Based on Evidential Neural Process Forecasting")
    print("=" * 80)
    
    # Check if model exists
    model_path = "CNP-model-save-name/saved_models/model_4000.pth"
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        print("The strategy will run with random weights for demonstration.")
        print("For real trading, please ensure you have a trained model.")
        print()
    
    # Configuration 1: Conservative strategy
    print("CONFIGURATION 1: Conservative Strategy")
    print("-" * 40)
    conservative_strategy = XAUUSDTradingStrategy(
        model_path=model_path,
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.005,  # 0.5% threshold (more conservative)
        max_position_size=0.5,  # Risk only 50% of capital
        device="cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    )
    
    print("Running conservative backtest...")
    conservative_report = conservative_strategy.run_backtest(
        data_file="datasets/Strategy_XAUUSD.csv",
        start_index=100,
        max_trades=200,  # Increased from 30 to 200
        step_size=3      # Reduced from 5 to 3 for more frequent checks
    )
    
    print("\nConservative Strategy Results:")
    print(f"  Final Capital: ${conservative_report['final_capital']:.2f}")
    print(f"  Total Return: {conservative_report['total_return_pct']:.2f}%")
    print(f"  Win Rate: {conservative_report['win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {conservative_report['max_drawdown_pct']:.2f}%")
    
    # Save results
    conservative_strategy.plot_results("trading_results/conservative")
    
    print("\n" + "=" * 50)
    
    # Configuration 2: Aggressive strategy
    print("CONFIGURATION 2: Aggressive Strategy")
    print("-" * 40)
    aggressive_strategy = XAUUSDTradingStrategy(
        model_path=model_path,
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.001,  # 0.1% threshold (more aggressive)
        max_position_size=0.8,  # Risk up to 80% of capital
        device="cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    )
    
    print("Running aggressive backtest...")
    aggressive_report = aggressive_strategy.run_backtest(
        data_file="datasets/Strategy_XAUUSD.csv",
        start_index=100,
        max_trades=300,  # Increased from 50 to 300
        step_size=2      # Reduced from 3 to 2 for more frequent checks
    )
    
    print("\nAggressive Strategy Results:")
    print(f"  Final Capital: ${aggressive_report['final_capital']:.2f}")
    print(f"  Total Return: {aggressive_report['total_return_pct']:.2f}%")
    print(f"  Win Rate: {aggressive_report['win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {aggressive_report['max_drawdown_pct']:.2f}%")
    
    # Save results
    aggressive_strategy.plot_results("trading_results/aggressive")
    
    # Comparison
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Conservative':<15} {'Aggressive':<15}")
    print("-" * 50)
    print(f"{'Total Return':<20} {conservative_report['total_return_pct']:>12.2f}% {aggressive_report['total_return_pct']:>12.2f}%")
    print(f"{'Win Rate':<20} {conservative_report['win_rate']*100:>12.1f}% {aggressive_report['win_rate']*100:>12.1f}%")
    print(f"{'Max Drawdown':<20} {conservative_report['max_drawdown_pct']:>12.2f}% {aggressive_report['max_drawdown_pct']:>12.2f}%")
    print(f"{'Total Trades':<20} {conservative_report['total_trades']:>12d} {aggressive_report['total_trades']:>12d}")
    
    print(f"\nResults saved to trading_results/ directory")
    
    return conservative_strategy, aggressive_strategy

def run_extended_data_demo():
    """
    Demonstrate creating and using extended datasets
    """
    print("\n" + "=" * 60)
    print("EXTENDED DATA DEMONSTRATION")
    print("=" * 60)
    
    from trading_data_manager import TradingDataManager
    
    # Create extended dataset
    data_manager = TradingDataManager()
    
    print("Creating extended dataset for longer backtests...")
    extended_df = data_manager.create_extended_dataset(
        output_file="datasets/Strategy_XAUUSD_extended.csv",
        num_samples=2000,  # Create 2000 samples for longer testing
        base_data_file="datasets/Strategy_XAUUSD.csv"
    )
    
    # Run strategy on extended data
    print("Running strategy on extended dataset...")
    extended_strategy = XAUUSDTradingStrategy(
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.002,
        max_position_size=0.7
    )
    
    extended_report = extended_strategy.run_backtest(
        data_file="datasets/Strategy_XAUUSD_extended.csv",
        start_index=200,
        max_trades=500,  # Increased from 100 to 500
        step_size=2      # Reduced from 4 to 2 for more frequent checks
    )
    
    print(f"\nExtended Data Results:")
    print(f"  Dataset Size: {len(extended_df)} samples")
    print(f"  Final Capital: ${extended_report['final_capital']:.2f}")
    print(f"  Total Return: {extended_report['total_return_pct']:.2f}%")
    print(f"  Total Trades: {extended_report['total_trades']}")
    
    extended_strategy.plot_results("trading_results/extended")
    
    return extended_strategy

def main():
    """Main function"""
    try:
        # Run strategy demonstrations
        conservative, aggressive = run_strategy_demo()
        
        # Optionally run extended data demo
        print("\nWould you like to test with extended data? (This creates a larger dataset)")
        # For automated demo, we'll skip the extended demo to save time
        # extended_strategy = run_extended_data_demo()
        
        print("\n" + "=" * 80)
        print("TRADING STRATEGY DEMONSTRATION COMPLETED")
        print("=" * 80)
        print("Key Features Demonstrated:")
        print("✓ Model-based price prediction using Evidential Neural Process")
        print("✓ Configurable trading parameters (thresholds, position sizes)")
        print("✓ Comprehensive backtesting with multiple strategies")
        print("✓ Detailed performance analytics and visualizations")
        print("✓ Risk management with position sizing and drawdown tracking")
        print("✓ Completely separate from original forecasting code")
        print()
        print("Next steps:")
        print("- Review results in trading_results/ directory")
        print("- Adjust strategy parameters based on performance")
        print("- Test with different datasets or time periods")
        print("- Implement live trading interface if desired")
        
    except Exception as e:
        print(f"Error running trading strategy: {e}")
        print("Please ensure all dependencies are installed and data files exist.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 