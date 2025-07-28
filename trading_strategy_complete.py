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
                 model_path="CNP-model-save-name/saved_models/model_4000.pth",
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
        
        # Trading state - UPDATED: Support multiple concurrent positions
        self.positions = []  # List of position dictionaries
        # Each position: {'type': 1/-1, 'size': float, 'entry_price': float, 'entry_index': int, 'id': str}
        self.next_position_id = 1  # Auto-incrementing position ID
        
        # Signal confirmation tracking
        self.recent_signals = []  # Track recent signals for confirmation
        self.required_confirmations = 2  # Require 2 consecutive signals
        
        # Legacy compatibility (for existing code that checks self.position)
        self.position = 0  # 0: no positions, 1: has long, -1: has short, 2: has both
        self.position_size = 0  # Total invested amount across all positions
        
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
        Check if there's a trading signal based on predictions with trend following
        
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
        
        # Simple trend following: get recent price trend
        trend_signal = self._get_trend_direction(current_index)
        
        # Generate signal based on significance threshold AND trend alignment
        if abs(price_change) > self.significance_threshold:
            raw_signal = 1 if price_change > 0 else -1
            
            # Only trade if signal aligns with trend (or no clear trend)
            if trend_signal == 0 or raw_signal == trend_signal:
                signal = raw_signal
                strength = abs(price_change)
                return signal, strength, predicted_price
        
        return 0, 0.0, predicted_price
    
    def _get_trend_direction(self, current_index):
        """
        Simple trend detection: compare current price to price 20 periods ago
        Returns: 1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        try:
            if current_index < 20:
                return 0  # Not enough data
            
            current_price = self.data_manager.get_price_at_index(current_index)
            past_price = self.data_manager.get_price_at_index(current_index - 20)
            
            trend_change = (current_price - past_price) / past_price
            
            if trend_change > 0.02:  # 2% uptrend
                return 1
            elif trend_change < -0.02:  # 2% downtrend
                return -1
            else:
                return 0  # Sideways
        except:
            return 0  # Default to no trend if error
    
    def _update_legacy_position_status(self):
        """Update legacy position variables for compatibility"""
        if not self.positions:
            self.position = 0
            self.position_size = 0
        else:
            # Calculate total position size
            self.position_size = sum(pos['size'] for pos in self.positions)
            
            # Determine position status
            has_long = any(pos['type'] == 1 for pos in self.positions)
            has_short = any(pos['type'] == -1 for pos in self.positions)
            
            if has_long and has_short:
                self.position = 2  # Both long and short
            elif has_long:
                self.position = 1  # Long only
            elif has_short:
                self.position = -1  # Short only
            else:
                self.position = 0  # No positions
    
    def _get_total_exposure(self):
        """Calculate total capital exposure across all positions"""
        return sum(pos['size'] for pos in self.positions)
    
    def _calculate_position_pnl(self, position, current_price):
        """Calculate current P&L for a position"""
        entry_price = position['entry_price']
        position_size = position['size']
        price_diff = current_price - entry_price
        shares = position_size / entry_price
        
        if position['type'] == 1:  # LONG
            return price_diff * shares
        else:  # SHORT
            return -price_diff * shares
    
    def _can_open_new_position(self, position_size):
        """Check if we can open a new position given capital constraints"""
        # SIMPLIFIED: Only allow ONE position at a time (much safer)
        if len(self.positions) >= 2:
            return False
            
        # Much smaller position sizes
        return position_size <= (self.current_capital * 0.3)  # Max 30% per trade
    
    def _should_close_positions(self, signal):
        """
        Determine which positions should be closed based on new signal
        
        Rules:
        - LONG → SHORT: Close ALL LONG positions
        - SHORT → LONG: Close ALL SHORT positions  
        - LONG → LONG: Keep all LONG positions (open another)
        - SHORT → SHORT: Keep all SHORT positions (open another)
        - Any → None: Keep all positions (hold)
        """
        positions_to_close = []
        
        if signal != 0:  # Only close positions when we have a signal
            # Check if we're switching directions
            has_long = any(pos['type'] == 1 for pos in self.positions)
            has_short = any(pos['type'] == -1 for pos in self.positions)
            
            if signal == 1 and has_short:  # Switching to LONG, close all SHORTs
                for i, pos in enumerate(self.positions):
                    if pos['type'] == -1:  # SHORT position
                        positions_to_close.append(i)
            elif signal == -1 and has_long:  # Switching to SHORT, close all LONGs
                for i, pos in enumerate(self.positions):
                    if pos['type'] == 1:  # LONG position
                        positions_to_close.append(i)
            # If signal matches existing positions (same direction), don't close anything
        
        return positions_to_close
    
    def execute_trade(self, signal, current_price, current_index, predicted_price, strength=1.0):
        """
        Execute a trading decision with support for multiple concurrent positions
        
        Args:
            signal: 1 for buy, -1 for sell, 0 for hold (no action)
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
            'positions_before': len(self.positions),
            'total_exposure_before': self._get_total_exposure(),
            'signal_strength': strength
        }
        
        positions_closed = 0
        position_opened = False
        
        # Determine current position status for action description
        has_long_before = any(pos['type'] == 1 for pos in self.positions)
        has_short_before = any(pos['type'] == -1 for pos in self.positions)
        
        # 1. Close positions based on rules (if any)
        positions_to_close = self._should_close_positions(signal)
        if positions_to_close:
            close_type = "LONG" if self.positions[positions_to_close[0]]['type'] == 1 else "SHORT"
            print(f"  🔄 Direction change detected: Closing all {close_type} positions ({len(positions_to_close)} positions)")
            
        for pos_index in reversed(positions_to_close):  # Reverse to maintain indices
            self._close_position_by_index(pos_index, current_price, current_index, trade_record)
            positions_closed += 1
        
        # 2. AGGRESSIVE risk management with profit taking and tight stops
        positions_to_force_close = []
        for i, pos in enumerate(self.positions):
            holding_period = current_index - pos['entry_index']
            current_pnl = self._calculate_position_pnl(pos, current_price)
            pnl_percent = current_pnl / pos['size']
            
            # MUCH tighter stop-loss (5%) and profit taking (10%)
            should_close = (
                holding_period > 20 or          # Close after 20 periods max
                pnl_percent < -0.05 or          # 5% stop loss
                pnl_percent > 0.10              # 10% profit taking
            )
            
            if should_close:
                positions_to_force_close.append(i)
        
        # Close forced positions
        for pos_index in reversed(positions_to_force_close):
            self._close_position_by_index(pos_index, current_price, current_index, trade_record)
            positions_closed += 1
        
        # 3. Signal confirmation and position opening
        # Track recent signals for confirmation
        self.recent_signals.append(signal)
        if len(self.recent_signals) > self.required_confirmations:
            self.recent_signals.pop(0)  # Keep only recent signals
        
        # Only trade if we have confirmed signals (reduce false positives)
        confirmed_signal = 0
        if len(self.recent_signals) >= self.required_confirmations:
            # Check if recent signals are consistent and non-zero
            if all(s == signal and s != 0 for s in self.recent_signals):
                confirmed_signal = signal
        
        if confirmed_signal != 0:
            position_size = self.current_capital * self.max_position_size
            if self._can_open_new_position(position_size):
                # Determine action type for clearer logging
                signal_type = "LONG" if confirmed_signal == 1 else "SHORT"
                action_desc = f"CONFIRMED_{signal_type}"
                
                self._open_new_position(confirmed_signal, current_price, current_index, position_size, trade_record, strength)
                trade_record['action_type'] = action_desc
                position_opened = True
            else:
                # Log that we wanted to trade but couldn't due to capital constraints
                signal_type = "LONG" if confirmed_signal == 1 else "SHORT"
                trade_record['action'] = f'BLOCKED_{signal_type}_CAPITAL'
        
        # 4. Update legacy position status
        self._update_legacy_position_status()
        
        # Record trade details
        trade_record.update({
            'capital_after': self.current_capital,
            'positions_after': len(self.positions),
            'total_exposure_after': self._get_total_exposure(),
            'positions_closed': positions_closed,
            'new_position_opened': position_opened,
            'active_positions': [{'type': p['type'], 'size': p['size'], 'id': p['id']} for p in self.positions]
        })
        
        # Only record if something actually happened
        if position_opened or positions_closed > 0 or signal != 0:
            self.trade_history.append(trade_record)
        
        self.capital_history.append(self.current_capital)
        self.position_history.append(self.position)
        
        return position_opened  # Return whether a new position was opened
    
    def _close_position_by_index(self, pos_index, current_price, current_index, trade_record):
        """Close a specific position by its index and calculate P&L"""
        if pos_index >= len(self.positions):
            return
        
        position = self.positions[pos_index]
        
        # Calculate P&L in dollars
        entry_price = position['entry_price']
        position_size = position['size']
        price_diff = current_price - entry_price
        
        # Calculate number of shares (or units) we could buy with position_size
        shares = position_size / entry_price
        
        if position['type'] == 1:  # Long position (buy)
            # P&L = price_difference * shares = (current_price - entry_price) * shares
            pnl = price_diff * shares
            action = f'CLOSE_LONG_{position["id"]}'
        else:  # Short position (sell)
            # P&L = negative_price_difference * shares = (entry_price - current_price) * shares
            pnl = -price_diff * shares
            action = f'CLOSE_SHORT_{position["id"]}'
        
        # Update capital
        self.current_capital += pnl
        
        # Create detailed trade record for this specific closure
        close_record = trade_record.copy()
        close_record.update({
            'action': action,
            'pnl': pnl,
            'return_pct': (pnl / position_size) * 100,
            'holding_period': current_index - position['entry_index'],
            'shares': shares,
            'price_diff': price_diff,
            'position_id': position['id'],
            'entry_price': entry_price,
            'position_size': position_size
        })
        
        # Add to trade history immediately for completed trades
        self.trade_history.append(close_record)
        
        # Remove position from active positions
        del self.positions[pos_index]
    
    def _open_new_position(self, signal, current_price, current_index, position_size, trade_record, strength):
        """Open a new position and add it to the positions list"""
        position_id = f"P{self.next_position_id}"
        self.next_position_id += 1
        
        # Create new position record
        new_position = {
            'type': signal,
            'size': position_size,
            'entry_price': current_price,
            'entry_index': current_index,
            'id': position_id,
            'strength': strength,
            'timestamp': self.data_manager.get_datetime_at_index(current_index)
        }
        
        # Add to positions list
        self.positions.append(new_position)
        
        # Update trade record
        if signal == 1:
            trade_record['action'] = f'OPEN_LONG_{position_id}'
        else:
            trade_record['action'] = f'OPEN_SHORT_{position_id}'
        
        trade_record.update({
            'position_id': position_id,
            'position_size': position_size,
            'position_fraction': position_size / self.current_capital,
            'entry_price': current_price
        })
    
    def run_backtest(self, 
                     data_file="datasets/Strategy_XAUUSD.csv", 
                     start_index=100, 
                     end_index=None,
                     max_trades=1000,
                     step_size=5):
        """
        Run backtesting on historical data
        
        Args:
            data_file: Path to historical data
            start_index: Starting index in the dataset
            end_index: Ending index in the dataset (None for full dataset)
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
        
        print(f"Starting backtest from index {start_index} to {end_index if end_index else 'end'}...")
        print(f"Target trades: {max_trades}")
        print()
        
        # Process data using the data manager iterator
        for data_point in self.data_manager.data_iterator(
            start_index=start_index, 
            end_index=end_index,   # NEW: Add end_index parameter
            step_size=step_size, 
            max_iterations=5000  # REDUCED: More conservative iteration limit
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
                
                # Execute trade if signal exists or we need to manage existing position
                should_trade = (signal != 0) or (self.position != 0)
                
                if should_trade:
                    position_opened = self.execute_trade(signal, current_price, current_index, 
                                                       predicted_price, strength)
                    
                    # Count only actual new positions opened (FIXED: Accurate trade counting)
                    if position_opened:
                        trades_executed += 1
                        action = "BUY" if signal == 1 else "SELL"
                        print(f"Trade {trades_executed:3d}: {action:4s} at ${current_price:7.2f} "
                              f"(Pred: ${predicted_price:7.2f}, Capital: ${self.current_capital:8.2f})")
                        
                        # Track prediction accuracy (FIXED: Check direction correctness)
                        expected_direction = 1 if predicted_price > current_price else -1
                        if expected_direction == signal:  # Prediction direction matches signal
                            successful_predictions += 1
                
            except Exception as e:
                print(f"Error at index {current_index}: {str(e)}")
                continue
        
        # Close any remaining positions
        if len(self.positions) > 0:
            final_price = self.data_manager.get_price_at_index(len(df) - 1)
            final_index = len(df) - 1
            
            print(f"Closing {len(self.positions)} remaining positions at ${final_price:.2f}")
            
            # Close each position individually
            positions_to_close = list(range(len(self.positions)))
            for pos_index in reversed(positions_to_close):  # Reverse to maintain indices
                trade_record = {
                    'timestamp': self.data_manager.get_datetime_at_index(final_index),
                    'index': final_index,
                    'price': final_price,
                    'predicted_price': final_price,
                    'capital_before': self.current_capital,
                    'signal_strength': 0
                }
                self._close_position_by_index(pos_index, final_price, final_index, trade_record)
            
            # Update legacy status
            self._update_legacy_position_status()
        
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
        model_path="CNP-model-save-name/saved_models/model_4000.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,  # As requested: look 5 steps ahead
        significance_threshold=0.002,  # 0.2% threshold for trading signals
        max_position_size=0.8,  # Risk max 80% of capital
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Run backtesting
    report = strategy.run_backtest(
        data_file="datasets/Strategy_XAUUSD.csv",
        start_index=100,
        max_trades=250,  # Increased for comprehensive testing
        step_size=2   # Check every 2 time steps for more opportunities
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