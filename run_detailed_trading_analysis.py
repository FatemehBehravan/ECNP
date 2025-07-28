#!/usr/bin/env python3
"""
Detailed Trading Analysis with Complete Trade History
Shows comprehensive trade-by-trade results for each strategy configuration
"""

import pandas as pd
import numpy as np
from datetime import datetime
from trading_strategy_complete import XAUUSDTradingStrategy

def analyze_trade_history(strategy, strategy_name):
    """
    Extract and analyze detailed trade history from a strategy
    
    Returns:
        DataFrame with detailed trade information
    """
    if not strategy.trade_history:
        return pd.DataFrame()
    
    # Convert trade history to DataFrame
    df_trades = pd.DataFrame(strategy.trade_history)
    
    # Filter to get only position opening and closing trades
    open_trades = df_trades[df_trades['action'].str.startswith('OPEN')].copy()
    close_trades = df_trades[df_trades['action'].str.startswith('CLOSE')].copy()
    
    detailed_trades = []
    
<<<<<<< HEAD
    # Get the last available close price from the data (for open positions)
    last_close_price = None
    if hasattr(strategy, 'data_manager') and hasattr(strategy.data_manager, 'original_prices'):
        if len(strategy.data_manager.original_prices) > 0:
            last_close_price = strategy.data_manager.original_prices[-1]
            print(f"  📊 Last available close price: ${last_close_price:.2f} (for open positions)")
    
=======
>>>>>>> 2dbd82a3f8dd7f34f49b806a4f1da3ba30ad40a2
    # Match opening and closing trades by position ID (NEW: Multiple position support)
    for i, open_trade in open_trades.iterrows():
        # Extract position ID from action (e.g., "OPEN_LONG_P1" -> "P1")
        if 'position_id' in open_trade:
            position_id = open_trade['position_id']
        else:
            # Fallback: extract from action string
            action_parts = open_trade['action'].split('_')
            position_id = action_parts[-1] if len(action_parts) > 2 else None
<<<<<<< HEAD
=======
        
        if position_id:
            # Find corresponding close trade with same position ID
            close_trade_candidates = close_trades[
                (close_trades.index > i) & 
                (close_trades['action'].str.contains(position_id, na=False))
            ]
        else:
            # Fallback to old logic for compatibility
            close_trade_candidates = close_trades[close_trades.index > i]
>>>>>>> 2dbd82a3f8dd7f34f49b806a4f1da3ba30ad40a2
        
        if position_id:
            # Find corresponding close trade with same position ID
            close_trade_candidates = close_trades[
                (close_trades.index > i) & 
                (close_trades['action'].str.contains(position_id, na=False))
            ]
        else:
            # Fallback to old logic for compatibility
            close_trade_candidates = close_trades[close_trades.index > i]
        
        # Check if we found a proper close trade
        if len(close_trade_candidates) > 0:
            close_trade = close_trade_candidates.iloc[0]
<<<<<<< HEAD
            exit_price = close_trade['price']
            closing_time = close_trade['timestamp']
            is_open_position = False
        else:
            # FIXED: No close trade found - this is an open position
            # Use the last available close price from the dataset
            if last_close_price is not None:
                exit_price = last_close_price
                closing_time = "OPEN_POSITION"
                is_open_position = True
                print(f"    🔍 Open position {position_id}: Using last close price ${exit_price:.2f}")
            else:
                print(f"    ❌ Open position {position_id}: No last close price available, skipping")
                continue
        
        # Calculate trade details
        entry_price = open_trade.get('entry_price', open_trade['price'])
        
        # Get the correct position size (NEW: Multiple position support)
        if not is_open_position and 'position_size' in close_trade:
            position_size = close_trade['position_size']
        else:
            position_size = open_trade.get('position_size', 0)
        
        # Calculate P&L properly
        if not is_open_position and 'pnl' in close_trade:
            # Use the P&L from close trade if available (for properly closed positions)
            pnl = close_trade['pnl']
            price_diff = close_trade.get('price_diff', exit_price - entry_price)
            shares = close_trade.get('shares', position_size / entry_price if entry_price > 0 else 0)
        else:
            # FIXED: Calculate P&L using correct formula for open positions
            price_diff = exit_price - entry_price
            shares = position_size / entry_price if entry_price > 0 else 0
            
            position_type = 'LONG' if 'LONG' in open_trade['action'] else 'SHORT'
            if position_type == 'LONG':
                pnl = price_diff * shares
            else:
                pnl = -price_diff * shares
        
        # Determine success/failure
        success = "SUCCESS" if pnl > 0 else "FAILED" if pnl < 0 else "BREAKEVEN"
        if is_open_position:
            success = f"{success}_OPEN"
        
        # Calculate position value (money actually invested)
        position_value = position_size  # This is already in dollars
        
        # Get position type from action
        position_type = 'LONG' if 'LONG' in open_trade['action'] else 'SHORT'
        
        trade_detail = {
            'Strategy': strategy_name,
            'Trade_Number': len(detailed_trades) + 1,
            'Position_ID': position_id if position_id else f"T{len(detailed_trades) + 1}",
            'Opening_Time': open_trade['timestamp'],
            'Closing_Time': closing_time,
            'Position_Type': position_type,
            'Trade_Rule': open_trade.get('action_type', 'UNKNOWN'),  # NEW: Show which rule was applied
            'Entry_Price': round(entry_price, 2),
            'Exit_Price': round(exit_price, 2),
            'Price_Difference': round(price_diff, 2),
            'PnL': round(pnl, 2),
            'Position_Size_USD': round(position_value, 2),
            'Return_Percent': round((pnl / position_value * 100) if position_value > 0 else 0, 2),
            'Holding_Period': close_trade.get('holding_period', 0) if not is_open_position else 'OPEN',
            'Result': success,
            'Signal_Strength': open_trade.get('signal_strength', 0),
            'Predicted_Price': open_trade.get('predicted_price', entry_price),
            'Shares': round(shares, 4),  # Number of shares bought
            'Open_Action': open_trade['action'],
            'Close_Action': close_trade['action'] if not is_open_position else 'OPEN_POSITION',
            'Debug_Info': f"ID:{position_id}, Rule:{open_trade.get('action_type', 'UNK')}, Type:{position_type}, Shares:{shares:.4f}, PnL:${pnl:.2f}, Status:{'OPEN' if is_open_position else 'CLOSED'}"
        }
        
        detailed_trades.append(trade_detail)
=======
            
            # Calculate trade details
            entry_price = open_trade.get('entry_price', open_trade['price'])
            exit_price = close_trade['price']
            
            # Get the correct position size (NEW: Multiple position support)
            position_size = open_trade.get('position_size', 
                                         close_trade.get('position_size', 0))
            
            # Use the P&L from close trade if available (NEW: More accurate)
            if 'pnl' in close_trade:
                pnl = close_trade['pnl']
                price_diff = close_trade.get('price_diff', exit_price - entry_price)
                shares = close_trade.get('shares', position_size / entry_price if entry_price > 0 else 0)
            else:
                # Fallback: Recalculate P&L using dollar-based formula
                price_diff = exit_price - entry_price
                shares = position_size / entry_price if entry_price > 0 else 0
                
                position_type = 'LONG' if 'LONG' in open_trade['action'] else 'SHORT'
                if position_type == 'LONG':
                    pnl = price_diff * shares
                else:
                    pnl = -price_diff * shares
            
            # Determine success/failure
            success = "SUCCESS" if pnl > 0 else "FAILED" if pnl < 0 else "BREAKEVEN"
            
            # Calculate position value (money actually invested)
            position_value = position_size  # This is already in dollars
            
            # Get position type from action
            position_type = 'LONG' if 'LONG' in open_trade['action'] else 'SHORT'
            
            trade_detail = {
                'Strategy': strategy_name,
                'Trade_Number': len(detailed_trades) + 1,
                'Position_ID': position_id if position_id else f"T{len(detailed_trades) + 1}",
                'Opening_Time': open_trade['timestamp'],
                'Closing_Time': close_trade['timestamp'],
                'Position_Type': position_type,
                'Trade_Rule': open_trade.get('action_type', 'UNKNOWN'),  # NEW: Show which rule was applied
                'Entry_Price': round(entry_price, 2),
                'Exit_Price': round(exit_price, 2),
                'Price_Difference': round(price_diff, 2),
                'PnL': round(pnl, 2),
                'Position_Size_USD': round(position_value, 2),
                'Return_Percent': round((pnl / position_value * 100) if position_value > 0 else 0, 2),
                'Holding_Period': close_trade.get('holding_period', 0),
                'Result': success,
                'Signal_Strength': open_trade.get('signal_strength', 0),
                'Predicted_Price': open_trade.get('predicted_price', entry_price),
                'Shares': round(shares, 4),  # Number of shares bought
                'Open_Action': open_trade['action'],
                'Close_Action': close_trade['action'],
                'Debug_Info': f"ID:{position_id}, Rule:{open_trade.get('action_type', 'UNK')}, Type:{position_type}, Shares:{shares:.4f}, PnL:${pnl:.2f}"
            }
            
            detailed_trades.append(trade_detail)
>>>>>>> 2dbd82a3f8dd7f34f49b806a4f1da3ba30ad40a2
    
    return pd.DataFrame(detailed_trades)

def analyze_signal_generation(strategy, config):
    """
    Analyze why aggressive strategies might have fewer trades than expected
    """
    if not strategy.trade_history:
        return {'signals_generated': 0, 'signals_blocked': 0, 'capital_depletion': False}
    
    df_trades = pd.DataFrame(strategy.trade_history)
    
    # Count different types of events
    open_trades = len(df_trades[df_trades['action'].str.startswith('OPEN')])
    close_trades = len(df_trades[df_trades['action'].str.startswith('CLOSE')])
    
    # Analyze capital evolution
    capital_history = df_trades['capital_after'].values
    initial_capital = df_trades['capital_before'].iloc[0]
    final_capital = capital_history[-1] if len(capital_history) > 0 else initial_capital
    
    # Check for capital depletion pattern
    capital_depletion = final_capital < initial_capital * 0.7  # Lost more than 30%
    
    # Estimate blocked signals (this is approximate)
    # If threshold is very low, we expect more signals, but positions might block them
    expected_signal_ratio = 0.008 / config['threshold']  # Relative to ultra conservative
    expected_trades = min(config['max_trades'], int(50 * expected_signal_ratio))  # Rough estimate
    
    analysis = {
        'strategy_name': config['name'],
        'threshold': config['threshold'],
        'expected_trades_estimate': expected_trades,
        'actual_open_trades': open_trades,
        'actual_close_trades': close_trades,
        'signal_efficiency': open_trades / max(expected_trades, 1),
        'capital_depletion': capital_depletion,
        'final_capital_ratio': final_capital / initial_capital,
        'avg_position_size': config['position_size']
    }
    
    return analysis

def run_detailed_strategy_analysis():
    """
    Run all strategies and extract detailed trade histories
    """
    print("=" * 100)
    print("DETAILED XAUUSD TRADING STRATEGY ANALYSIS")
    print("Complete Trade-by-Trade History")
    print("🆕 NEW: Conservative Single-Position Trading with Risk Management")
    print("=" * 100)
    print("📋 ENHANCED FEATURES:")
    print("  🔒 SINGLE POSITION: Only one position at a time (much safer)")
    print("  ✅ SIGNAL CONFIRMATION: Requires 2+ consecutive signals")
    print("  📈 TREND FOLLOWING: Only trades with 20-period trend direction")
    print("  💸 TIGHT STOPS: 5% stop-loss, 10% profit-taking")
    print("  ⏰ TIME LIMITS: Auto-close positions after 20 periods")
    print("  💰 SMALL SIZES: 5-20% position sizes (much safer)")
    print("  🛡️ CAPITAL PROTECTION: Max 30% capital per trade")
    print("")
    print("📊 RISK MANAGEMENT:")
    print("  • No over-leveraging with single position limit")
    print("  • False signal reduction via confirmation")
    print("  • Trend alignment prevents counter-trend trades")
    print("  • Quick profit-taking and loss-cutting")
    print("=" * 100)
    
    # Strategy configurations (UPDATED: Realistic thresholds based on model predictions)
    # Debug showed avg predicted change = 14.81%, so using meaningful thresholds
    configurations = [
        {
            "name": "Ultra Conservative",
            "threshold": 0.10,  # 10% threshold - Only trade on extreme predictions
            "position_size": 0.1,  # 10% max position
            "max_trades": 5000,
            "step_size": 2
        },
        {
            "name": "Conservative", 
            "threshold": 0.07,  # 7% threshold - Trade on very strong predictions
            "position_size": 0.15,  # 15% max position
            "max_trades": 5000,
            "step_size": 2
        },
        {
            "name": "Moderate",
            "threshold": 0.04,  # 4% threshold - Trade on strong predictions
            "position_size": 0.2,  # 20% max position
            "max_trades": 5000,
            "step_size": 2
        },
        {
            "name": "Aggressive",
            "threshold": 0.01,  # 1% threshold - Trade on above-average predictions
            "position_size": 0.25,  # 25% max position
            "max_trades": 5000,
            "step_size": 2
        },
        {
            "name": "Ultra Aggressive",
            "threshold": 0.005,  # 0.5% threshold - Trade on most predictions
            "position_size": 0.3,  # 30% max position
            "max_trades": 5000,
            "step_size": 2
        }
    ]
    
    all_trade_histories = []
    strategy_summaries = []
    
    for config in configurations:
        print(f"\n{'='*30} {config['name']} Strategy {'='*30}")
        print(f"Threshold: {config['threshold']*100:.2f}% | Position Size: {config['position_size']*100:.0f}% | Target: {config['max_trades']} trades")
        print("-" * 90)
        
        # Initialize and run strategy
        strategy = XAUUSDTradingStrategy(
            model_path="CNP-model-save-name/saved_models/model_4000.pth",
            initial_capital=1000.0,
            prediction_lookforward=5,
            significance_threshold=config['threshold'],
            max_position_size=config['position_size'],
            device="cuda"
        )
        
        # Run backtest with detailed tracking
        print(f"  🔄 Starting backtest with detailed debugging...")
        
        # Add debugging variables
        debug_info = {
            'predictions_attempted': 0,
            'predictions_successful': 0,
            'signals_generated': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'trades_opened': 0,
            'trades_closed': 0,
            'errors': 0
        }
        
        report = strategy.run_backtest(
            data_file="datasets/Strategy_XAUUSD.csv",
<<<<<<< HEAD
            start_index=100,  # CHANGED: Start from index 100
            end_index=1000,   # NEW: End at index 1000
=======
            start_index=1000,  # CHANGED: Start from index 100
            end_index=5000,   # NEW: End at index 1000
>>>>>>> 2dbd82a3f8dd7f34f49b806a4f1da3ba30ad40a2
            max_trades=config['max_trades'],
            step_size=config['step_size']
        )
        
        # Analyze what happened during backtest
        print(f"  🔍 DEBUGGING BACKTEST EXECUTION:")
        if strategy.trade_history:
            opened_actions = [t for t in strategy.trade_history if 'OPEN' in t['action']]
            closed_actions = [t for t in strategy.trade_history if 'CLOSE' in t['action']]
            blocked_actions = [t for t in strategy.trade_history if 'BLOCKED' in t.get('action', '')]
            
            print(f"    Total trade records: {len(strategy.trade_history)}")
            print(f"    OPEN actions: {len(opened_actions)}")
            print(f"    CLOSE actions: {len(closed_actions)}")
            print(f"    BLOCKED actions: {len(blocked_actions)} (capital constraints)")
            print(f"    Final position: {strategy.position} (0=None, 1=Long, -1=Short)")
            print(f"    Final capital: ${strategy.current_capital:.2f}")
            print(f"    Active positions: {len(strategy.positions)}")
            print(f"    Total exposure: ${strategy._get_total_exposure():.2f} ({strategy._get_total_exposure()/strategy.current_capital*100:.1f}% of capital)")
            
            # Check if positions are being held indefinitely
            if len(opened_actions) > len(closed_actions):
                print(f"    ⚠️  OPEN POSITIONS DETECTED: {len(opened_actions) - len(closed_actions)} unclosed positions")
                print(f"    This may explain why no 'completed' trades are reported.")
            
            # Check for capital constraints
            if len(blocked_actions) > 0:
                print(f"    🚫 CAPITAL CONSTRAINTS: {len(blocked_actions)} trades blocked due to insufficient capital")
                
            # Show sample of trade history with rule types
            if len(strategy.trade_history) > 0:
                print(f"    First few actions:")
                for i, trade in enumerate(strategy.trade_history[:5]):
                    action = trade['action']
                    action_type = trade.get('action_type', 'UNKNOWN')
                    
                    # Choose symbol based on action type
                    if "OPEN_LONG" in action:
                        if action_type == "ADD_LONG":
                            symbol = "📈➕"  # Add long
                        elif action_type == "SWITCH_TO_LONG":
                            symbol = "🔄📈"  # Switch to long
                        else:
                            symbol = "📈"   # Open long
                    elif "OPEN_SHORT" in action:
                        if action_type == "ADD_SHORT":
                            symbol = "📉➕"  # Add short
                        elif action_type == "SWITCH_TO_SHORT":
                            symbol = "🔄📉"  # Switch to short
                        else:
                            symbol = "📉"   # Open short
                    elif "CLOSE" in action:
                        symbol = "❌"    # Close
                    elif "BLOCKED" in action:
                        symbol = "🚫"    # Blocked
                    else:
                        symbol = "⏸️"     # Other
                    
                    print(f"      {i+1}. {symbol} {action} at ${trade['price']:.2f}")
                    if action_type != 'UNKNOWN':
                        print(f"          Rule: {action_type} (Strength: {trade.get('signal_strength', 0):.4f})")
        else:
            print(f"    ❌ NO TRADE HISTORY GENERATED!")
            print(f"    🔍 POSSIBLE REASONS:")
            print(f"       1. Model predictions are failing (returns None)")
            print(f"       2. No signals meet the threshold ({config['threshold']*100:.2f}%)")
            print(f"       3. Data loading/iteration issues")
            print(f"       4. Hardcoded iteration limits hit (max_iterations)")
            print(f"       5. Capital constraints preventing trades")
        
        # Debug: Analyze why aggressive strategies have fewer trades
        signal_analysis = analyze_signal_generation(strategy, config)
        
        # Extract detailed trade history
        trade_history_df = analyze_trade_history(strategy, config['name'])
        
        if not trade_history_df.empty:
            all_trade_histories.append(trade_history_df)
            
            # Print summary
            successful_trades = len(trade_history_df[trade_history_df['Result'] == 'SUCCESS'])
            failed_trades = len(trade_history_df[trade_history_df['Result'] == 'FAILED'])
            total_trades = len(trade_history_df)
            
            avg_pnl_success = trade_history_df[trade_history_df['Result'] == 'SUCCESS']['PnL'].mean() if successful_trades > 0 else 0
            avg_pnl_failed = trade_history_df[trade_history_df['Result'] == 'FAILED']['PnL'].mean() if failed_trades > 0 else 0
            
            print(f"\n📊 TRADE SUMMARY:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Successful: {successful_trades} ({successful_trades/total_trades*100:.1f}%)")
            print(f"  Failed: {failed_trades} ({failed_trades/total_trades*100:.1f}%)")
            print(f"  Avg Profit per Winning Trade: ${avg_pnl_success:.2f}")
            print(f"  Avg Loss per Losing Trade: ${avg_pnl_failed:.2f}")
            print(f"  Final Capital: ${report['final_capital']:.2f}")
            print(f"  Total Return: {report['total_return_pct']:.2f}%")
            
            # Show signal analysis
            print(f"\n🎯 SIGNAL ANALYSIS:")
            print(f"  Expected Trades (estimate): {signal_analysis['expected_trades_estimate']}")
            print(f"  Actual Trades: {signal_analysis['actual_open_trades']}")
            print(f"  Signal Efficiency: {signal_analysis['signal_efficiency']:.2f}")
            print(f"  Capital Ratio: {signal_analysis['final_capital_ratio']:.3f}")
            if signal_analysis['capital_depletion']:
                print(f"  ⚠️  CAPITAL DEPLETION DETECTED!")
            
            # Show first few trades as example
            # print(f"\n📋 FIRST 5 TRADES:")
            # print(trade_history_df.head().to_string(index=False))
            
            # Store strategy summary
            strategy_summaries.append({
                'Strategy': config['name'],
                'Total_Trades': total_trades,
                'Successful_Trades': successful_trades,
                'Failed_Trades': failed_trades,
                'Win_Rate': successful_trades/total_trades*100 if total_trades > 0 else 0,
                'Avg_Profit_Win': avg_pnl_success,
                'Avg_Loss_Lose': avg_pnl_failed,
                'Final_Capital': report['final_capital'],
                'Total_Return_Pct': report['total_return_pct'],
                'Max_Drawdown_Pct': report['max_drawdown_pct']
            })
        else:
            print("No completed trades found for this strategy.")
    
    # Combine all trade histories
    if all_trade_histories:
        combined_trades = pd.concat(all_trade_histories, ignore_index=True)
        
        # Save detailed trade histories
        print(f"\n{'='*100}")
        print("SAVING DETAILED RESULTS")
        print("="*100)
        
        # Save individual strategy trade histories
        for strategy_name in [config['name'] for config in configurations]:
            strategy_trades = combined_trades[combined_trades['Strategy'] == strategy_name]
            if not strategy_trades.empty:
                filename = f"trading_results/detailed_analysis/{strategy_name.lower().replace(' ', '_')}_trades.csv"
                import os
                os.makedirs("trading_results/detailed_analysis", exist_ok=True)
                strategy_trades.to_csv(filename, index=False)
                print(f"✅ Saved {len(strategy_trades)} trades for {strategy_name}: {filename}")
        
        # Save combined results
        combined_trades.to_csv("trading_results/detailed_analysis/all_strategies_trades.csv", index=False)
        print(f"✅ Saved combined trade history: trading_results/detailed_analysis/all_strategies_trades.csv")
        
        # Create and save strategy comparison
        summary_df = pd.DataFrame(strategy_summaries)
        summary_df.to_csv("trading_results/detailed_analysis/strategy_comparison.csv", index=False)
        print(f"✅ Saved strategy comparison: trading_results/detailed_analysis/strategy_comparison.csv")
        
        # Print final comparison
        print(f"\n{'='*100}")
        print("FINAL STRATEGY COMPARISON")
        print("="*100)
        print(summary_df.to_string(index=False))
        

        
        # Print signal analysis summary
        print(f"\n{'='*100}")
        print("🚨 SIGNAL GENERATION ANALYSIS - WHY AGGRESSIVE STRATEGIES HAVE FEWER TRADES")
        print("="*100)
        
        signal_summary = []
        for strategy_name in [config['name'] for config in configurations]:
            strategy_trades = combined_trades[combined_trades['Strategy'] == strategy_name]
            if not strategy_trades.empty:
                config_match = next(c for c in configurations if c['name'] == strategy_name)
                signal_data = {
                    'Strategy': strategy_name,
                    'Threshold_%': config_match['threshold'] * 100,
                    'Position_Size_%': config_match['position_size'] * 100,
                    'Actual_Trades': len(strategy_trades),
                    'Expected_Sensitivity': f"{0.8 / config_match['threshold']:.1f}x",
                    'Avg_Holding_Period': strategy_trades['Holding_Period'].mean(),
                    'Success_Rate_%': len(strategy_trades[strategy_trades['Result'] == 'SUCCESS']) / len(strategy_trades) * 100
                }
                signal_summary.append(signal_data)
        
        signal_df = pd.DataFrame(signal_summary)
        print(signal_df.to_string(index=False))
        
        print(f"\n🔍 POSSIBLE REASONS FOR FEWER AGGRESSIVE TRADES:")
        print("1. 💰 Capital Depletion: Higher position sizes (70-80%) lose money faster")
        print("2. 🔒 Position Blocking: Positions stay open longer, blocking new signals")
        print("3. 📉 Poor Performance: Aggressive strategies might be less profitable")
        print("4. 🎯 Model Limitations: Very small thresholds might not generate reliable signals")
        print("5. ⏱️  Longer Holding Periods: Positions held longer = fewer new opportunities")
        
        return combined_trades, summary_df
    else:
        print("No trade data collected from any strategy.")
        return None, None

def display_trade_details_by_strategy(combined_trades):
    """
    Display detailed trade information organized by strategy
    """
    if combined_trades is None or combined_trades.empty:
        print("No trade data available.")
        return
    
    print(f"\n{'='*120}")
    print("DETAILED TRADE HISTORY BY STRATEGY")
    print("="*120)
    
    for strategy in combined_trades['Strategy'].unique():
        strategy_trades = combined_trades[combined_trades['Strategy'] == strategy]
        
        print(f"\n🔸 {strategy.upper()} STRATEGY - {len(strategy_trades)} TRADES")
        print("-" * 120)
        
        # Format for better display (NEW: Include Position_ID and Trade_Rule for multiple position tracking)
        display_columns = [
            'Trade_Number', 'Position_ID', 'Trade_Rule', 'Opening_Time', 'Position_Type', 'Entry_Price', 
            'Exit_Price', 'Price_Difference', 'PnL', 'Position_Size_USD', 
            'Return_Percent', 'Result'
        ]
        
        display_df = strategy_trades[display_columns].copy()
        
        # Format timestamps to be more readable
        display_df['Opening_Time'] = pd.to_datetime(display_df['Opening_Time']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Color code results for terminal display
        for idx, row in display_df.iterrows():
            result_symbol = "✅" if row['Result'] == 'SUCCESS' else "❌" if row['Result'] == 'FAILED' else "➖"
            display_df.at[idx, 'Result'] = f"{result_symbol} {row['Result']}"
        
        print(display_df.to_string(index=False))
        
        # Strategy statistics
        total_pnl = strategy_trades['PnL'].sum()
        avg_return = strategy_trades['Return_Percent'].mean()
        max_win = strategy_trades['PnL'].max()
        max_loss = strategy_trades['PnL'].min()
        
        print(f"\n📈 STRATEGY STATISTICS:")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Average Return per Trade: {avg_return:.2f}%")
        print(f"  Largest Win: ${max_win:.2f}")
        print(f"  Largest Loss: ${max_loss:.2f}")



def test_single_prediction_sequence():
    """
    Test a single prediction sequence to see exactly what the model predicts
    """
    print(f"\n{'='*100}")
    print("🔬 DETAILED SINGLE PREDICTION ANALYSIS")
    print("="*100)
    
    strategy = XAUUSDTradingStrategy(
        model_path="CNP-model-save-name/saved_models/model_4000.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.005,  # 0.5%
        max_position_size=0.1,
        device="cuda"
    )
    
    # Load data into the data manager (CRITICAL: This was missing!)
    strategy.data_manager.load_extended_data("datasets/Strategy_XAUUSD.csv")
    
    # Get one data point
    data_iterator = strategy.data_manager.data_iterator(start_index=100, end_index=None, step_size=1, max_iterations=1)
    data_point = next(data_iterator)
    
    current_index = data_point['index']
    current_price = data_point['current_price']
    
    print(f"Testing at index: {current_index}")
    print(f"Current price: ${current_price:.2f}")
    
    # Get detailed predictions
    predictions = strategy.predict_future_prices(current_index)
    
    if predictions is not None:
        print(f"\n📈 RAW MODEL PREDICTIONS:")
        for i, pred_price in enumerate(predictions['predictions'][:10]):
            change = (pred_price - current_price) / current_price * 100
            print(f"  Step {i:2d}: ${pred_price:8.2f} (Change: {change:+6.3f}%)")
        
        # Test different thresholds on this single prediction
        lookforward_price = predictions['predictions'][strategy.prediction_lookforward]
        actual_change = (lookforward_price - current_price) / current_price
        
        print(f"\n🎯 THRESHOLD TESTING (Lookforward {strategy.prediction_lookforward}):")
        print(f"   Predicted price: ${lookforward_price:.2f}")
        print(f"   Actual change: {actual_change*100:.4f}%")
        
        test_thresholds = [0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
        for thresh in test_thresholds:
            would_signal = abs(actual_change) > thresh
            signal_type = "BUY" if actual_change > 0 else "SELL" if actual_change < 0 else "HOLD"
            status = "✅ SIGNAL" if would_signal else "❌ NO SIGNAL"
            print(f"   {thresh*100:5.2f}% threshold: {status} ({signal_type})")

def quick_model_data_test():
    """
    Quick test to verify model and data are working before running full analysis
    """
    print(f"\n{'='*60}")
    print("🧪 QUICK MODEL & DATA TEST")
    print("="*60)
    
    try:
        # Test model loading and prediction
        print("1. Testing model loading...")
        strategy = XAUUSDTradingStrategy(
            model_path="CNP-model-save-name/saved_models/model_4000.pth",
            initial_capital=1000.0,
            prediction_lookforward=5,
            significance_threshold=0.05,  # 5%
            max_position_size=0.1,
            device="cuda"
        )
        print("   ✅ Model loaded successfully")
        
        # Test data loading
        print("2. Testing data loading...")
        df = strategy.data_manager.load_extended_data("datasets/Strategy_XAUUSD.csv")
        print(f"   ✅ Data loaded: {len(df)} records")
        
        # Test single prediction
        print("3. Testing single prediction...")
        data_iterator = strategy.data_manager.data_iterator(start_index=100, end_index=None, step_size=1, max_iterations=1)
        data_point = next(data_iterator)
        
        predictions = strategy.predict_future_prices(data_point['index'])
        if predictions is not None:
            print(f"   ✅ Prediction successful: {len(predictions['predictions'])} predicted prices")
            
            # Test signal generation
            signal, strength, pred_price = strategy.check_trading_signal(
                predictions, data_point['current_price'], data_point['index']
            )
            print(f"   ✅ Signal generated: {signal} (strength: {strength:.4f})")
            print(f"   Current price: ${data_point['current_price']:.2f}")
            print(f"   Predicted price: ${pred_price:.2f}")
            print(f"   Price change: {((pred_price - data_point['current_price']) / data_point['current_price'] * 100):+.4f}%")
            
        else:
            print("   ❌ Prediction FAILED")
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("="*60)

def main():
    """Main execution function"""
    print("Choose analysis mode:")
    print("1. Full Strategy Analysis (original functionality)")
    print("2. Single Prediction Analysis")
    print("3. Full Analysis + Single Prediction Analysis")
    print("4. Quick Model & Data Test (Debug)")
    
    # For automation, you can change this
    choice = input("Enter choice (1-4): ").strip()
    
    combined_trades, summary_df = None, None
    
    # Run Quick Test
    if choice == "4":
        quick_model_data_test()
        return None, None
    
    # Run Full Strategy Analysis
    if choice in ["1", "3"]:
        print("\n" + "="*100)
        print("RUNNING FULL STRATEGY ANALYSIS")
        print("="*100)
        
        # Run detailed analysis
        combined_trades, summary_df = run_detailed_strategy_analysis()
        
        if combined_trades is not None:
            # Display detailed trade information
            display_trade_details_by_strategy(combined_trades)
            
            print(f"\n{'='*120}")
            print("✅ FULL ANALYSIS COMPLETE")
            print("="*120)
            print("📁 All detailed trade histories saved to: trading_results/detailed_analysis/")
            print("📊 Files generated:")
            print("   • Individual strategy trade logs (CSV)")
            print("   • Combined trade history (CSV)")
            print("   • Strategy performance comparison (CSV)")
    
    # Run Single Prediction Analysis
    if choice in ["2", "3"]:
        test_single_prediction_sequence()
    
    # Return appropriate results
    if choice == "1":
        return combined_trades, summary_df
    elif choice == "2":
        return None, None  # Just ran single prediction
    elif choice == "3":
        print("\n🎯 All analysis complete! Use the generated files for further investigation.")
        return combined_trades, summary_df
    else:
        print("Invalid choice.")
        return None, None

if __name__ == "__main__":
    trades, summary = main() 