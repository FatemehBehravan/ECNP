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
    
    # Match opening and closing trades
    for i, open_trade in open_trades.iterrows():
        # Find corresponding close trade
        close_trade_candidates = close_trades[close_trades.index > i]
        
        if len(close_trade_candidates) > 0:
            close_trade = close_trade_candidates.iloc[0]
            
            # Calculate trade details
            entry_price = open_trade['price']
            exit_price = close_trade['price']
            
            # Get the correct position size (use actual_position_size if available)
            position_size = open_trade.get('actual_position_size', 
                                         open_trade.get('position_size_after', 0))
            
            # Recalculate P&L correctly using dollar-based formula (don't trust the stored pnl)
            price_diff = exit_price - entry_price
            position_type = 'LONG' if 'LONG' in open_trade['action'] else 'SHORT'
            
            # Calculate number of shares (or units) we could buy with position_size
            shares = position_size / entry_price if entry_price > 0 else 0
            
            if position_type == 'LONG':
                # For LONG: profit when exit_price > entry_price
                pnl = price_diff * shares
            else:
                # For SHORT: profit when exit_price < entry_price  
                pnl = -price_diff * shares
            
            # Determine success/failure
            success = "SUCCESS" if pnl > 0 else "FAILED" if pnl < 0 else "BREAKEVEN"
            
            # Calculate position value (money actually invested)
            position_value = position_size  # This is already in dollars
            
            trade_detail = {
                'Strategy': strategy_name,
                'Trade_Number': len(detailed_trades) + 1,
                'Opening_Time': open_trade['timestamp'],
                'Closing_Time': close_trade['timestamp'],
                'Position_Type': position_type,
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
                'Original_PnL': close_trade.get('pnl', 0),  # For debugging
                'Shares': round(shares, 4),  # Number of shares bought
                'Debug_Info': f"Type:{position_type}, Shares:{shares:.4f}, PriceDiff:${price_diff:.2f}, PnL:${pnl:.2f}"
            }
            
            detailed_trades.append(trade_detail)
    
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
    print("=" * 100)
    
    # Strategy configurations (with fixed position sizes as requested)
    configurations = [
        {
            "name": "Ultra Conservative",
            "threshold": 0.008,  # 0.8% threshold
            "position_size": 0.1,  # 10% max position
            "max_trades": 1000,
            "step_size": 2
        },
        {
            "name": "Conservative", 
            "threshold": 0.005,  # 0.5% threshold
            "position_size": 0.15,  # 15% max position
            "max_trades": 1000,
            "step_size": 2
        },
        {
            "name": "Moderate",
            "threshold": 0.003,  # 0.3% threshold
            "position_size": 0.2,  # 20% max position
            "max_trades": 1000,
            "step_size": 2
        },
        {
            "name": "Aggressive",
            "threshold": 0.001,  # 0.1% threshold
            "position_size": 0.25,  # 25% max position
            "max_trades": 1000,
            "step_size": 2
        },
        {
            "name": "Ultra Aggressive",
            "threshold": 0.0005,  # 0.05% threshold
            "position_size": 0.3,  # 30% max position
            "max_trades": 1000,
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
        report = strategy.run_backtest(
            data_file="datasets/Strategy_XAUUSD.csv",
            start_index=50,
            max_trades=config['max_trades'],
            step_size=config['step_size']
        )
        
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
            print(f"\n📋 FIRST 5 TRADES:")
            print(trade_history_df.head().to_string(index=False))
            
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
        
        # Format for better display
        display_columns = [
            'Trade_Number', 'Opening_Time', 'Position_Type', 'Entry_Price', 
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



def debug_aggressive_strategy_issue():
    """
    Specifically debug why aggressive strategies have fewer trades
    """
    print(f"\n{'='*100}")
    print("🔍 DEEP DIVE: AGGRESSIVE STRATEGY TRADE COUNT ISSUE")
    print("="*100)
    
    # Compare two strategies directly
    configs_to_compare = [
        {
            "name": "Conservative",
            "threshold": 0.005,
            "position_size": 0.15,  # Updated to new 15%
            "color": "🟢"
        },
        {
            "name": "Ultra Aggressive", 
            "threshold": 0.0005,
            "position_size": 0.3,  # Updated to new 30%
            "color": "🔴"
        }
    ]
    
    for config in configs_to_compare:
        print(f"\n{config['color']} TESTING {config['name'].upper()} STRATEGY")
        print("-" * 60)
        
        strategy = XAUUSDTradingStrategy(
            model_path="CNP-model-save-name/saved_models/model_4000.pth",
            initial_capital=1000.0,
            prediction_lookforward=5,
            significance_threshold=config['threshold'],
            max_position_size=config['position_size'],
            device="cuda"
        )
        
        # Track signals and decisions
        original_execute_trade = strategy.execute_trade
        signal_log = []
        
        def tracked_execute_trade(signal, current_price, current_index, predicted_price, strength=1.0):
            signal_log.append({
                'index': current_index,
                'signal': signal,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'strength': strength,
                'position_before': strategy.position,
                'capital_before': strategy.current_capital
            })
            return original_execute_trade(signal, current_price, current_index, predicted_price, strength)
        
        strategy.execute_trade = tracked_execute_trade
        
        # Run limited backtest
        report = strategy.run_backtest(
            data_file="datasets/Strategy_XAUUSD.csv",
            start_index=50,
            max_trades=100,  # Smaller for detailed analysis
            step_size=2
        )
        
        # Analyze signals
        signal_df = pd.DataFrame(signal_log)
        if not signal_df.empty:
            buy_signals = len(signal_df[signal_df['signal'] == 1])
            sell_signals = len(signal_df[signal_df['signal'] == -1])
            hold_signals = len(signal_df[signal_df['signal'] == 0])
            blocked_signals = len(signal_df[(signal_df['signal'] != 0) & (signal_df['position_before'] != 0)])
            
            print(f"  📊 SIGNAL BREAKDOWN:")
            print(f"    Total Signal Checks: {len(signal_df)}")
            print(f"    Buy Signals: {buy_signals}")
            print(f"    Sell Signals: {sell_signals}") 
            print(f"    Hold/No Signal: {hold_signals}")
            print(f"    Blocked Signals (position open): {blocked_signals}")
            print(f"    Actual Trades Executed: {report['total_trades']}")
            print(f"    Final Capital: ${report['final_capital']:.2f}")
            
            # Check signal strength distribution
            strong_signals = len(signal_df[signal_df['strength'] > config['threshold']])
            print(f"    Strong Signals (>{config['threshold']*100:.2f}%): {strong_signals}")
            
    print(f"\n🎯 KEY INSIGHTS:")
    print("• If Ultra Aggressive shows many 'Blocked Signals', positions stay open too long")
    print("• If Ultra Aggressive has fewer 'Strong Signals', model doesn't generate small movements")
    print("• If Ultra Aggressive loses capital faster, fewer trades possible due to reduced position sizes")

def main():
    """Main execution function"""
    print("Choose analysis mode:")
    print("1. Full Strategy Analysis (original functionality)")
    print("2. Aggressive Strategy Trade Count Debug")
    print("3. Full Analysis + Aggressive Debug")
    
    # For automation, you can change this
    choice = input("Enter choice (1-3): ").strip()
    
    combined_trades, summary_df = None, None
    
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
    
    # Run Aggressive Strategy Debug  
    if choice in ["2", "3"]:
        debug_aggressive_strategy_issue()
    
    # Return appropriate results
    if choice == "1":
        return combined_trades, summary_df
    elif choice == "2":
        return None, None  # Just ran aggressive debug
    elif choice == "3":
        print("\n🎯 All analysis complete! Use the generated files for further investigation.")
        return combined_trades, summary_df
    else:
        print("Invalid choice.")
        return None, None

if __name__ == "__main__":
    trades, summary = main() 