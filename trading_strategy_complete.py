import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the existing model and data components
from models.np_complete_models import Transformer_Evd_Model
from trading_data_manager import TradingDataManager
from utilFiles.save_load_files_models import load_model
from utilFiles.get_args import the_args
from models.shared_model_detail import *

class XAUUSDTradingStrategy:
    """
    Complete trading strategy for XAUUSD forecasting using Evidential Neural Process
    
    Strategy Logic:
    - When y_pred[i+lookforward] > current_price + threshold: BUY
    - When y_pred[i+lookforward] < current_price - threshold: SELL
    - P&L calculation: (target_y_orig[i+lookforward] - target_y_orig[i]) for buy positions
    """
    
    def __init__(self, 
                 model_path="CNP-model-save-name/saved_models/best_model.pth",
                 initial_capital=1000.0,
                 prediction_lookforward=5,
                 significance_threshold=0.002,  # 0.2% price change threshold
                 max_position_size=0.8,  # Maximum 80% of capital per trade
                 device="cuda"):
        """
        Initialize the trading strategy
        
        Args:
            model_path: Path to the trained forecasting model
            initial_capital: Starting capital in USD
            prediction_lookforward: How many steps ahead to look (default 5 as requested)
            significance_threshold: Minimum price change to trigger trade (as fraction)
            max_position_size: Maximum fraction of capital to risk per trade
            device: Computing device (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.prediction_lookforward = prediction_lookforward
        self.significance_threshold = significance_threshold
        self.max_position_size = max_position_size
        
        # Trading state
        self.position = 0  # 0: no position, 1: long (buy), -1: short (sell)
        self.position_size = 0  # Amount invested in current position
        self.entry_price = 0   # Price at which position was opened
        self.entry_index = 0   # Index at which position was opened
        
        # Performance tracking
        self.trade_history = []
        self.capital_history = [initial_capital]
        self.position_history = [0]
        self.daily_returns = []
        
        # Load the trained model
        print("Loading forecasting model...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize data manager
        self.data_manager = TradingDataManager(device=self.device)
        
        print(f"Trading Strategy initialized:")
        print(f"  Initial Capital: ${initial_capital}")
        print(f"  Prediction Lookforward: {prediction_lookforward} steps")
        print(f"  Significance Threshold: {significance_threshold*100:.2f}%")
        print(f"  Device: {self.device}")
        
    def _load_model(self, model_path):
        """Load the trained forecasting model"""
        args = the_args()
        
        # Create model architecture matching the original
        model = Transformer_Evd_Model(
            latent_encoder_sizes,
            determministic_encoder_sizes,
            decoder_sizes + [4*args.channels],  # NIG parameters
            args,
            attention,
        ).to(self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"  Model loaded from: {model_path}")
        else:
            print(f"  Warning: Model file {model_path} not found. Using random weights.")
            
        return model
    
    def predict_future_prices(self, current_index):
        """
        Generate price predictions using the trained model
        
        Args:
            current_index: Current position in the dataset
            
        Returns:
            Dictionary with predictions and metadata
        """
        with torch.no_grad():
            # Get context data (past sequences for prediction)
            context_data = self.data_manager.get_context_data(current_index, context_length=30)
            if context_data is None:
                return None
                
            # Get target data (future sequences to predict)
            target_data = self.data_manager.get_target_data(current_index, target_length=20)
            if target_data is None:
                return None
            
            # Create model input
            model_input = self.data_manager.create_model_input(context_data, target_data)
            if model_input is None:
                return None
            
            # Get model predictions
            query, target_y = model_input.query, model_input.target_y
            dist, _, _, _, mu, v, alpha, beta = self.model(query, None)
            
            # Convert predictions back to original scale
            mu_cpu = mu.cpu().numpy()
            predictions_original = []
            
            for i in range(mu_cpu.shape[1]):  # For each predicted sequence
                for j in range(mu_cpu.shape[2]):  # For each time step in sequence
                    scaled_price = mu_cpu[0, i, j, 0]
                    original_price = self.data_manager.inverse_transform_price(scaled_price)
                    predictions_original.append(original_price)
            
            # Calculate uncertainty measures
            epistemic = (beta / (v * (alpha - 1))).cpu().numpy()
            aleatoric = (beta / (alpha - 1)).cpu().numpy()
            
            return {
                'predictions': np.array(predictions_original),
                'predictions_raw': mu_cpu,
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric,
                'context_data': context_data,
                'target_data': target_data
            }
    
    def check_trading_signal(self, predictions, current_price, current_index):
        """
        Check if there's a trading signal based on predictions
        
        Strategy: Compare y_pred[i+lookforward] with current_price
        
        Args:
            predictions: Model predictions dictionary
            current_price: Current market price
            current_index: Current position in dataset
            
        Returns:
            tuple: (signal, strength, predicted_price)
                signal: 1 for buy, -1 for sell, 0 for hold
                strength: Signal strength (price change magnitude)
                predicted_price: Predicted price at lookforward step
        """
        if predictions is None or len(predictions['predictions']) <= self.prediction_lookforward:
            return 0, 0.0, current_price
        
        # Get predicted price at lookforward step
        predicted_price = predictions['predictions'][self.prediction_lookforward]
        
        # Calculate relative price change
        price_change = (predicted_price - current_price) / current_price
        
        # Generate signal based on significance threshold
        if abs(price_change) > self.significance_threshold:
            signal = 1 if price_change > 0 else -1  # 1 for buy, -1 for sell
            strength = abs(price_change)
            return signal, strength, predicted_price
        
        return 0, 0.0, predicted_price
    
    def execute_trade(self, signal, current_price, current_index, predicted_price, strength=1.0):
        """
        Execute a trading decision based on the strategy
        
        Args:
            signal: 1 for buy, -1 for sell, 0 for close position
            current_price: Current market price
            current_index: Current position in dataset
            predicted_price: Predicted future price
            strength: Signal strength (affects position size)
        """
        timestamp = self.data_manager.get_datetime_at_index(current_index)
        
        trade_record = {
            'timestamp': timestamp,
            'index': current_index,
            'action': '',
            'price': current_price,
            'predicted_price': predicted_price,
            'capital_before': self.current_capital,
            'position_before': self.position,
            'position_size_before': self.position_size,
            'signal_strength': strength
        }
        
        # Close existing position if we have one and signal is opposite or neutral
        if self.position != 0 and (signal != self.position or signal == 0):
            self._close_position(current_price, current_index, trade_record)
        
        # Open new position if signal is strong enough
        if signal != 0 and self.position == 0:
            self._open_position(signal, current_price, current_index, trade_record, strength)
        
        # Record trade
        trade_record.update({
            'capital_after': self.current_capital,
            'position_after': self.position,
            'position_size_after': self.position_size
        })
        
        self.trade_history.append(trade_record)
        self.capital_history.append(self.current_capital)
        self.position_history.append(self.position)
    
    def _close_position(self, current_price, current_index, trade_record):
        """Close the current position and calculate P&L"""
        if self.position == 0:
            return
        
        # Calculate P&L using the formula: (target_y_orig[i+5] - target_y_orig[i])
        # where i is the entry point and i+5 is current closing point
        entry_price = self.entry_price
        price_diff = current_price - entry_price
        
        if self.position == 1:  # Long position (buy)
            # P&L = (current_price - entry_price) * position_size_ratio
            pnl = price_diff / entry_price * self.position_size
            trade_record['action'] = 'CLOSE_LONG'
        else:  # Short position (sell)
            # P&L = (entry_price - current_price) * position_size_ratio  
            pnl = -price_diff / entry_price * self.position_size
            trade_record['action'] = 'CLOSE_SHORT'
        
        # Update capital
        self.current_capital += pnl
        trade_record['pnl'] = pnl
        trade_record['return_pct'] = pnl / self.position_size * 100
        trade_record['holding_period'] = current_index - self.entry_index
        
        # Reset position
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.entry_index = 0
    
    def _open_position(self, signal, current_price, current_index, trade_record, strength):
        """Open a new position"""
        # Calculate position size based on signal strength and available capital
        # More confident predictions get larger position sizes
        position_fraction = min(self.max_position_size, 0.2 + strength * 2)  # 20% to max_position_size
        self.position_size = self.current_capital * position_fraction
        
        self.position = signal
        self.entry_price = current_price
        self.entry_index = current_index
        
        if signal == 1:
            trade_record['action'] = 'OPEN_LONG'
        else:
            trade_record['action'] = 'OPEN_SHORT'
        
        trade_record['position_fraction'] = position_fraction
    
    def run_backtest(self, 
                     data_file="datasets/XAUUSD.csv", 
                     start_index=100, 
                     max_trades=1000,
                     step_size=5):
        """
        Run backtesting on historical data
        
        Args:
            data_file: Path to historical data
            start_index: Starting index in the dataset
            max_trades: Maximum number of trades to execute
            step_size: Step size for moving through data
        """
        print("=" * 60)
        print("STARTING XAUUSD TRADING BACKTEST")
        print("=" * 60)
        
        # Load historical data
        print("Loading historical data...")
        df = self.data_manager.load_extended_data(data_file)
        print(f"Data loaded: {len(df)} records")
        
        trades_executed = 0
        successful_predictions = 0
        total_predictions = 0
        
        print(f"Starting backtest from index {start_index}...")
        print(f"Target trades: {max_trades}")
        print()
        
        # Process data using the data manager iterator
        for data_point in self.data_manager.data_iterator(
            start_index=start_index, 
            step_size=step_size, 
            max_iterations=2000
        ):
            if trades_executed >= max_trades:
                break
                
            try:
                current_index = data_point['index']
                current_price = data_point['current_price']
                current_time = data_point['datetime']
                
                # Generate predictions
                predictions = self.predict_future_prices(current_index)
                if predictions is None:
                    continue
                
                total_predictions += 1
                
                # Check for trading signal
                signal, strength, predicted_price = self.check_trading_signal(
                    predictions, current_price, current_index
                )
                
                # Execute trade if signal exists or we need to close existing position
                should_trade = (signal != 0) or (self.position != 0)
                
                if should_trade:
                    self.execute_trade(signal, current_price, current_index, 
                                     predicted_price, strength)
                    
                    if signal != 0:  # New position opened
                        trades_executed += 1
                        action = "BUY" if signal == 1 else "SELL"
                        print(f"Trade {trades_executed:3d}: {action:4s} at ${current_price:7.2f} "
                              f"(Pred: ${predicted_price:7.2f}, Capital: ${self.current_capital:8.2f})")
                        
                        # Track prediction accuracy
                        if abs(predicted_price - current_price) > current_price * 0.001:  # >0.1% change
                            successful_predictions += 1
                
            except Exception as e:
                print(f"Error at index {current_index}: {str(e)}")
                continue
        
        # Close any remaining position
        if self.position != 0:
            final_price = self.data_manager.get_price_at_index(len(df) - 1)
            final_index = len(df) - 1
            self.execute_trade(0, final_price, final_index, final_price)
            print(f"Final position closed at ${final_price:.2f}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETED")
        print("=" * 60)
        print(f"Final Capital: ${self.current_capital:.2f}")
        print(f"Total Return: {((self.current_capital - self.initial_capital) / self.initial_capital * 100):.2f}%")
        print(f"Trades Executed: {trades_executed}")
        print(f"Prediction Accuracy: {successful_predictions}/{total_predictions} ({100*successful_predictions/max(total_predictions,1):.1f}%)")
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive trading performance report"""
        if not self.trade_history:
            return {"error": "No trades executed"}
        
        df_trades = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Trade analysis
        closed_trades = df_trades[df_trades['action'].str.startswith('CLOSE')]
        if len(closed_trades) > 0:
            profitable_trades = len(closed_trades[closed_trades['pnl'] > 0])
            total_trades = len(closed_trades)
            win_rate = profitable_trades / total_trades
            
            avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
            avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean() if (total_trades - profitable_trades) > 0 else 0
            
            max_win = closed_trades['pnl'].max()
            max_loss = closed_trades['pnl'].min()
        else:
            profitable_trades = total_trades = win_rate = avg_win = avg_loss = max_win = max_loss = 0
        
        # Calculate maximum drawdown
        peak_capital = self.initial_capital
        max_drawdown = 0
        for capital in self.capital_history:
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        # Risk metrics
        returns = [(self.capital_history[i] - self.capital_history[i-1]) / self.capital_history[i-1] 
                  for i in range(1, len(self.capital_history))]
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
        sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
        
        report = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        return report
    
    def plot_results(self, save_path="trading_results"):
        """Generate comprehensive trading performance plots"""
        if not self.capital_history:
            print("No trading data to plot")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Capital Evolution
        ax1.plot(self.capital_history, linewidth=2, color='blue')
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', 
                   label=f'Initial Capital (${self.initial_capital})')
        ax1.set_title('Capital Evolution Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Position History
        positions = np.array(self.position_history)
        colors = ['red' if p == -1 else 'green' if p == 1 else 'gray' for p in positions]
        ax2.scatter(range(len(positions)), positions, c=colors, alpha=0.6, s=20)
        ax2.set_title('Position History', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        if len(self.capital_history) > 1:
            returns = [(self.capital_history[i] - self.capital_history[i-1]) / self.capital_history[i-1] * 100
                      for i in range(1, len(self.capital_history))]
            ax3.hist(returns, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
            ax3.set_title('Returns Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Return
        cumulative_returns = [(cap - self.initial_capital) / self.initial_capital * 100 
                             for cap in self.capital_history]
        ax4.plot(cumulative_returns, linewidth=2, color='green')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_title('Cumulative Return (%)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/trading_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed trade log
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            df_trades.to_csv(f"{save_path}/detailed_trade_log.csv", index=False)
        
        # Save summary report
        report = self.generate_report()
        with open(f"{save_path}/performance_summary.txt", 'w') as f:
            f.write("XAUUSD Trading Strategy Performance Report\n")
            f.write("=" * 50 + "\n\n")
            for key, value in report.items():
                if 'pct' in key or key in ['win_rate', 'total_return']:
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f}%\n")
                elif isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        print(f"Results and plots saved to {save_path}/")

def main():
    """Main function to run the XAUUSD trading strategy"""
    print("XAUUSD Trading Strategy")
    print("Based on Evidential Neural Process Forecasting")
    print("=" * 70)
    
    # Initialize trading strategy with your requested parameters
    strategy = XAUUSDTradingStrategy(
        model_path="CNP-model-save-name/saved_models/best_model.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,  # As requested: look 5 steps ahead
        significance_threshold=0.002,  # 0.2% threshold for trading signals
        max_position_size=0.8,  # Risk max 80% of capital
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Run backtesting
    report = strategy.run_backtest(
        data_file="datasets/XAUUSD.csv",
        start_index=100,
        max_trades=50,  # Reasonable number for initial testing
        step_size=3   # Check every 3 time steps for efficiency
    )
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED PERFORMANCE REPORT")
    print("=" * 60)
    
    for key, value in report.items():
        if 'pct' in key or key in ['win_rate', 'total_return']:
            print(f"{key.replace('_', ' ').title():.<30} {value:.2f}%")
        elif isinstance(value, float):
            print(f"{key.replace('_', ' ').title():.<30} {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title():.<30} {value}")
    
    # Generate comprehensive plots and save results
    strategy.plot_results("trading_results")
    
    return strategy, report

if __name__ == "__main__":
    strategy, report = main() 