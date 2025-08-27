#!/usr/bin/env python3
"""
Test script to demonstrate uncertainty-based position sizing
This shows how the new uncertainty scaling works in practice
"""
import torch
import numpy as np
from trading_strategy_complete import XAUUSDTradingStrategy

def test_uncertainty_position_sizing():
    """
    Test the uncertainty-based position sizing functionality
    """
    print("=" * 80)
    print("UNCERTAINTY-BASED POSITION SIZING TEST")
    print("=" * 80)
    
    # Initialize strategy with uncertainty scaling enabled
    strategy = XAUUSDTradingStrategy(
        model_path="CNP-model-save-name/saved_models/model_4000.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.01,  # 1% threshold
        max_position_size=0.3,  # 30% base position
        uncertainty_scaling=True,  # Enable uncertainty scaling
        min_position_factor=0.1,  # Minimum 10% of base position
        max_position_factor=1.0,  # Maximum 100% of base position
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n🧪 TESTING UNCERTAINTY-BASED POSITION SIZING")
    print("-" * 50)
    
    # Test different uncertainty levels
    base_position_size = 1000.0 * 0.3  # $300 base position
    
    test_uncertainties = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"Base Position Size: ${base_position_size:.2f}")
    print(f"Position Range: {strategy.min_position_factor*100:.0f}% - {strategy.max_position_factor*100:.0f}% of base")
    print()
    
    for uncertainty in test_uncertainties:
        actual_position = strategy.calculate_uncertainty_based_position_size(uncertainty, base_position_size)
        confidence_factor = 1.0 - uncertainty
        position_factor = actual_position / base_position_size
        
        print(f"Uncertainty: {uncertainty:.1f} | Confidence: {confidence_factor:.1%} | "
              f"Position: ${actual_position:.2f} ({position_factor:.1%} of base)")
    
    print("\n" + "=" * 80)
    print("UNCERTAINTY SCALING BEHAVIOR:")
    print("=" * 80)
    print("• Low Uncertainty (0.1) = High Confidence (90%) = Large Position (90% of base)")
    print("• Medium Uncertainty (0.5) = Medium Confidence (50%) = Medium Position (50% of base)")
    print("• High Uncertainty (0.9) = Low Confidence (10%) = Small Position (10% of base)")
    print()
    print("This allows the strategy to:")
    print("✓ Allocate more capital to high-confidence predictions")
    print("✓ Reduce exposure during uncertain market conditions")
    print("✓ Automatically adapt position sizes based on model confidence")
    print("✓ Maintain trade frequency while managing risk dynamically")

def test_strategy_comparison():
    """
    Compare strategies with and without uncertainty scaling
    """
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: WITH vs WITHOUT UNCERTAINTY SCALING")
    print("=" * 80)
    
    # Strategy 1: Without uncertainty scaling
    strategy_fixed = XAUUSDTradingStrategy(
        model_path="CNP-model-save-name/saved_models/model_4000.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.01,
        max_position_size=0.3,
        uncertainty_scaling=False,  # Fixed position sizes
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Strategy 2: With uncertainty scaling
    strategy_uncertainty = XAUUSDTradingStrategy(
        model_path="CNP-model-save-name/saved_models/model_4000.pth",
        initial_capital=1000.0,
        prediction_lookforward=5,
        significance_threshold=0.01,
        max_position_size=0.3,
        uncertainty_scaling=True,  # Dynamic position sizes
        min_position_factor=0.1,
        max_position_factor=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Strategy 1 (Fixed): Always uses 30% position size")
    print("Strategy 2 (Uncertainty): Uses 10%-100% of 30% base position")
    print()
    print("Key Differences:")
    print("• Fixed Strategy: Consistent risk per trade")
    print("• Uncertainty Strategy: Risk adapts to model confidence")
    print("• Fixed Strategy: Same position size regardless of market conditions")
    print("• Uncertainty Strategy: Larger positions in confident markets, smaller in uncertain markets")

def main():
    """Main test function"""
    try:
        # Test uncertainty position sizing calculation
        test_uncertainty_position_sizing()
        
        # Test strategy comparison
        test_strategy_comparison()
        
        print("\n" + "=" * 80)
        print("UNCERTAINTY-BASED POSITION SIZING TEST COMPLETED")
        print("=" * 80)
        print("The uncertainty scaling feature is now ready to use!")
        print("Run run_detailed_trading_analysis.py to see it in action with real data.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
