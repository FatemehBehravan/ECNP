#!/usr/bin/env python3
"""
Extensive Trading Strategy Testing
Run comprehensive backtests with high numbers of trades for thorough evaluation
"""

from trading_strategy_complete import XAUUSDTradingStrategy

def run_extensive_trading():
    """
    Run extensive trading tests with high trade counts for comprehensive analysis
    """
    print("=" * 80)
    print("EXTENSIVE XAUUSD TRADING STRATEGY TESTING")
    print("High-Volume Trading Analysis")
    print("=" * 80)
    
    configurations = [
        {
            "name": "Ultra Conservative",
            "threshold": 0.008,  # 0.8% threshold
            "position_size": 0.4,  # 40% max position
            "max_trades": 150,
            "step_size": 4
        },
        {
            "name": "Conservative", 
            "threshold": 0.005,  # 0.5% threshold
            "position_size": 0.5,  # 50% max position
            "max_trades": 300,
            "step_size": 3
        },
        {
            "name": "Moderate",
            "threshold": 0.003,  # 0.3% threshold
            "position_size": 0.6,  # 60% max position
            "max_trades": 500,
            "step_size": 2
        },
        {
            "name": "Aggressive",
            "threshold": 0.001,  # 0.1% threshold
            "position_size": 0.7,  # 70% max position
            "max_trades": 750,
            "step_size": 2
        },
        {
            "name": "Ultra Aggressive",
            "threshold": 0.0005,  # 0.05% threshold
            "position_size": 0.8,  # 80% max position
            "max_trades": 1000,
            "step_size": 1
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n{'='*20} {config['name']} Strategy {'='*20}")
        print(f"Threshold: {config['threshold']*100:.2f}%")
        print(f"Max Position: {config['position_size']*100:.0f}%") 
        print(f"Target Trades: {config['max_trades']}")
        print(f"Step Size: {config['step_size']}")
        print("-" * 60)
        
        # Initialize strategy
        strategy = XAUUSDTradingStrategy(
            model_path="CNP-model-save-name/saved_models/model_4000.pth",
            initial_capital=1000.0,
            prediction_lookforward=5,
            significance_threshold=config['threshold'],
            max_position_size=config['position_size'],
            device="cuda"
        )
        
        # Run backtest
        report = strategy.run_backtest(
            data_file="datasets/Strategy_XAUUSD.csv",
            start_index=50,  # Start earlier for more data
            max_trades=config['max_trades'],
            step_size=config['step_size']
        )
        
        # Store results
        results[config['name']] = {
            'strategy': strategy,
            'report': report,
            'config': config
        }
        
        # Save individual results
        save_path = f"trading_results/extensive/{config['name'].lower().replace(' ', '_')}"
        strategy.plot_results(save_path)
        
        print(f"\nResults for {config['name']} Strategy:")
        print(f"  Final Capital: ${report['final_capital']:.2f}")
        print(f"  Total Return: {report['total_return_pct']:.2f}%")
        print(f"  Win Rate: {report['win_rate']*100:.1f}%")
        print(f"  Total Trades: {report['total_trades']}")
        print(f"  Max Drawdown: {report['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {report.get('sharpe_ratio', 0):.3f}")
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 80)
    
    print(f"{'Strategy':<18} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Drawdown':<10} {'Sharpe':<8}")
    print("-" * 80)
    
    for name, result in results.items():
        report = result['report']
        print(f"{name:<18} {report['total_return_pct']:>7.2f}% {report['total_trades']:>6d} "
              f"{report['win_rate']*100:>8.1f}% {report['max_drawdown_pct']:>8.2f}% "
              f"{report.get('sharpe_ratio', 0):>6.3f}")
    
    # Find best performing strategies
    best_return = max(results.items(), key=lambda x: x[1]['report']['total_return_pct'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['report'].get('sharpe_ratio', 0))
    best_winrate = max(results.items(), key=lambda x: x[1]['report']['win_rate'])
    
    print("\n" + "=" * 50)
    print("TOP PERFORMERS")
    print("=" * 50)
    print(f"Best Return: {best_return[0]} ({best_return[1]['report']['total_return_pct']:.2f}%)")
    print(f"Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['report'].get('sharpe_ratio', 0):.3f})")
    print(f"Best Win Rate: {best_winrate[0]} ({best_winrate[1]['report']['win_rate']*100:.1f}%)")
    
    print(f"\nDetailed results saved to trading_results/extensive/ directory")
    
    return results

def run_marathon_test():
    """
    Run a marathon test with maximum trade volume
    """
    print("\n" + "=" * 60)
    print("MARATHON TRADING TEST")
    print("Maximum Trade Volume Analysis")
    print("=" * 60)
    
    strategy = XAUUSDTradingStrategy(
        model_path="CNP-model-save-name/saved_models/model_4000.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.0008,  # Very sensitive
        max_position_size=0.6,  # Moderate risk
        device="cuda"
    )
    
    print("Running marathon test...")
    print("Target: 2000+ trades")
    print("This may take several minutes...")
    
    report = strategy.run_backtest(
        data_file="datasets/Strategy_XAUUSD.csv",
        start_index=25,  # Start very early
        max_trades=2000,  # Very high target
        step_size=1  # Check every single data point
    )
    
    strategy.plot_results("trading_results/marathon")
    
    print(f"\nMarathon Test Results:")
    print(f"Final Capital: ${report['final_capital']:.2f}")
    print(f"Total Return: {report['total_return_pct']:.2f}%")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate']*100:.1f}%")
    print(f"Max Drawdown: {report['max_drawdown_pct']:.2f}%")
    
    return strategy, report

def main():
    """Main function"""
    print("Choose testing mode:")
    print("1. Extensive Testing (5 configurations, up to 1000 trades each)")
    print("2. Marathon Test (Single strategy, 2000+ trades)")
    print("3. Both")
    
    # For automation, run extensive testing
    choice = "1"  # You can change this or make it interactive
    
    if choice in ["1", "3"]:
        results = run_extensive_trading()
    
    if choice in ["2", "3"]:
        marathon_strategy, marathon_report = run_marathon_test()
    
    print("\n" + "=" * 80)
    print("EXTENSIVE TRADING ANALYSIS COMPLETED")
    print("=" * 80)
    print("All results saved with detailed analytics and visualizations")
    
    return True

if __name__ == "__main__":
    main() 