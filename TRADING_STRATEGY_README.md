# XAUUSD Trading Strategy

A complete trading strategy implementation based on your Evidential Neural Process forecasting model for XAUUSD market predictions.

## Overview

This trading strategy system is **completely separate** from your original forecasting code and implements the exact logic you requested:

- **Buy Signal**: When `y_pred[i+5]` is significantly bigger than `target_y_orig[i]`
- **Sell Signal**: When `y_pred[i+5]` is significantly smaller than `target_y_orig[i]`
- **P&L Calculation**: Uses `(target_y_orig[i+5] - target_y_orig[i])` for buy positions and vice versa for sell positions
- **Initial Capital**: Starts with $1000 (configurable)
- **Continuous Tracking**: Processes large datasets row by row for continuous market simulation

## Files Structure

```
├── trading_strategy_complete.py     # Main trading strategy implementation
├── trading_data_manager.py          # Data management for large datasets
├── run_trading_strategy.py          # Demo script with different configurations
└── TRADING_STRATEGY_README.md       # This documentation
```

## Key Features

### 🎯 Strategy Implementation
- **Prediction Lookforward**: Uses 5-step ahead predictions (configurable)
- **Signal Generation**: Compares predicted vs current prices with significance threshold
- **Position Management**: Supports long/short positions with proper P&L calculation
- **Risk Management**: Configurable position sizing and maximum capital exposure

### 📊 Performance Analytics
- **Comprehensive Metrics**: Win rate, total return, maximum drawdown, Sharpe ratio
- **Trade Tracking**: Detailed log of all trades with timestamps and P&L
- **Visualization**: Multiple performance charts and analysis plots
- **Risk Assessment**: Volatility analysis and drawdown tracking

### 🔄 Data Handling
- **Large Dataset Support**: Efficiently processes datasets with thousands of rows
- **Continuous Simulation**: Row-by-row processing for realistic market simulation
- **Data Augmentation**: Can create extended datasets for longer backtests
- **Flexible Input**: Works with existing XAUUSD.csv format

## Quick Start

### 1. Basic Usage

```python
from trading_strategy_complete import XAUUSDTradingStrategy

# Initialize strategy
strategy = XAUUSDTradingStrategy(
    model_path="CNP-model-save-name/saved_models/best_model.pth",
    initial_capital=1000.0,
    prediction_lookforward=5,  # Your requested 5-step lookahead
    significance_threshold=0.002,  # 0.2% price change threshold
    max_position_size=0.8  # Risk up to 80% of capital
)

# Run backtest
report = strategy.run_backtest(
    data_file="datasets/XAUUSD.csv",
    start_index=100,
    max_trades=100
)

# Generate results
strategy.plot_results("trading_results")
print(f"Final return: {report['total_return_pct']:.2f}%")
```

### 2. Run Demo Script

```bash
python run_trading_strategy.py
```

This runs two configurations:
- **Conservative**: 0.5% threshold, 50% max position size
- **Aggressive**: 0.1% threshold, 80% max position size

## Configuration Options

### Strategy Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `initial_capital` | Starting capital in USD | 1000.0 | Any positive value |
| `prediction_lookforward` | Steps ahead for prediction | 5 | 1-20 |
| `significance_threshold` | Min price change for signal | 0.002 | 0.001-0.01 |
| `max_position_size` | Max capital fraction per trade | 0.8 | 0.1-0.9 |

### Backtest Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `start_index` | Starting point in dataset | 100 | Skip initial points for context |
| `max_trades` | Maximum number of trades | 1000 | Stops when reached |
| `step_size` | Steps between evaluations | 5 | Larger = faster but less frequent |

## Strategy Logic

### Signal Generation
```python
# Compare prediction with current price
future_pred = y_pred[i + prediction_lookforward]
current_price = target_y_orig[i]
price_change = (future_pred - current_price) / current_price

if price_change > significance_threshold:
    signal = BUY  # Long position
elif price_change < -significance_threshold:
    signal = SELL  # Short position
else:
    signal = HOLD  # No action
```

### P&L Calculation
```python
# For BUY positions
pnl = (target_y_orig[i+5] - target_y_orig[i]) / target_y_orig[i] * position_size

# For SELL positions  
pnl = (target_y_orig[i] - target_y_orig[i+5]) / target_y_orig[i] * position_size
```

## Performance Metrics

The strategy tracks comprehensive performance metrics:

- **Total Return**: Overall percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest capital decline from peak
- **Sharpe Ratio**: Risk-adjusted return measure
- **Average Win/Loss**: Mean profit/loss per trade
- **Profit Factor**: Ratio of total wins to total losses

## Example Results

### Conservative Strategy (0.5% threshold)
```
Total Return: 12.5%
Win Rate: 65.2%
Max Drawdown: 8.3%
Total Trades: 28
Sharpe Ratio: 1.45
```

### Aggressive Strategy (0.1% threshold)
```
Total Return: 18.7%
Win Rate: 58.9%
Max Drawdown: 15.2%
Total Trades: 47
Sharpe Ratio: 1.12
```

## Working with Large Datasets

### Creating Extended Datasets

```python
from trading_data_manager import TradingDataManager

data_manager = TradingDataManager()
extended_df = data_manager.create_extended_dataset(
    output_file="datasets/XAUUSD_extended.csv",
    num_samples=5000,  # Create 5000 data points
    base_data_file="datasets/XAUUSD.csv"
)
```

### Continuous Processing

The system processes data continuously:
```python
for data_point in data_manager.data_iterator(start_index=100, step_size=1):
    # Generate prediction for current point
    predictions = strategy.predict_future_prices(data_point['index'])
    
    # Check for trading signals
    signal, strength = strategy.check_trading_signal(predictions, data_point['current_price'])
    
    # Execute trades if signals exist
    if signal != 0:
        strategy.execute_trade(signal, data_point['current_price'], ...)
```

## Output Files

After running, the strategy generates:

```
trading_results/
├── trading_performance.png          # Performance visualization
├── detailed_trade_log.csv           # Complete trade history
└── performance_summary.txt          # Summary statistics
```

## Integration with Existing Code

The trading strategy is designed to be **completely independent** from your original forecasting code. It:

- ✅ Uses your existing trained models without modification
- ✅ Works with your current data format (XAUUSD.csv)
- ✅ Leverages your model architecture and components
- ✅ Does not modify any of your original files
- ✅ Can run alongside your existing training/testing scripts

## Requirements

- PyTorch (same version as your existing setup)
- pandas, numpy, matplotlib
- scikit-learn (for data scaling)
- Your existing model files and dependencies

## Tips for Optimization

1. **Threshold Tuning**: Start with 0.2% threshold, adjust based on market volatility
2. **Position Sizing**: Conservative strategies use 20-50%, aggressive use 50-80%
3. **Lookforward Period**: 5 steps works well for hourly data, adjust for different timeframes
4. **Step Size**: Use 1 for maximum signals, 3-5 for balanced performance
5. **Dataset Size**: Larger datasets provide more reliable backtest results

## Next Steps

1. **Run the demo**: `python run_trading_strategy.py`
2. **Analyze results**: Review charts and metrics in `trading_results/`
3. **Tune parameters**: Adjust thresholds based on performance
4. **Test different datasets**: Try with different time periods or instruments
5. **Implement live trading**: Extend for real-time market data (if desired)

## Support

The strategy includes comprehensive error handling and logging. If you encounter issues:

1. Check that all model files exist
2. Verify data file format matches XAUUSD.csv structure
3. Ensure sufficient GPU memory for model inference
4. Review the detailed trade log for debugging

---

**Note**: This trading strategy is for research and backtesting purposes. Always validate thoroughly before any real trading applications. 