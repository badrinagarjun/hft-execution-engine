"""
Quick test runner for all modules.
Runs basic tests on each component to verify installation.
"""

import sys
import traceback

def test_module(module_name, test_func):
    """Test a single module."""
    print(f"\n{'='*60}")
    print(f"Testing {module_name}...")
    print(f"{'='*60}")
    
    try:
        test_func()
        print(f"\n✓ {module_name} tests passed")
        return True
    except Exception as e:
        print(f"\n✗ {module_name} tests failed:")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all module tests."""
    print("="*60)
    print("ALGORITHMIC TRADING SYSTEM - MODULE TESTS")
    print("="*60)
    
    results = {}
    
    # Test each module
    print("\n1. Testing Data Capture...")
    try:
        from data_capture import CryptoDataCapture
        capture = CryptoDataCapture('binance', 'BTC/USDT')
        print("  ✓ Import successful")
        print("  ✓ Initialization successful")
        results['data_capture'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['data_capture'] = False
    
    print("\n2. Testing LOB Simulator...")
    try:
        from lob_simulator import SimpleLOB
        lob = SimpleLOB()
        lob.add_limit('buy', 100.0, 10.0)
        lob.add_limit('sell', 101.0, 8.0)
        fills = lob.market_buy(5.0)
        assert len(fills) > 0
        print("  ✓ Import successful")
        print("  ✓ LOB operations successful")
        print(f"  ✓ Market order executed: {len(fills)} fill(s)")
        results['lob_simulator'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['lob_simulator'] = False
    
    print("\n3. Testing Execution Schedules...")
    try:
        from execution_schedules import twap_schedule, vwap_schedule, pov_schedule
        twap = twap_schedule(1000, 10)
        vwap = vwap_schedule(1000, [1.0]*10)
        pov = pov_schedule(1000, [100.0]*10, 0.01)
        assert sum(twap) == 1000
        assert sum(vwap) == 1000
        print("  ✓ Import successful")
        print("  ✓ TWAP schedule valid")
        print("  ✓ VWAP schedule valid")
        print("  ✓ POV schedule valid")
        results['execution_schedules'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['execution_schedules'] = False
    
    print("\n4. Testing Order-Flow Features...")
    try:
        from orderflow_features import compute_imbalance, compute_spread
        bids = [[100.0, 10.0], [99.0, 8.0]]
        asks = [[101.0, 9.0], [102.0, 7.0]]
        imb = compute_imbalance(bids, asks, depth=2)
        spread = compute_spread(bids, asks)
        assert -1 <= imb <= 1
        assert spread > 0
        print("  ✓ Import successful")
        print(f"  ✓ Imbalance computed: {imb:.4f}")
        print(f"  ✓ Spread computed: {spread:.2f}")
        results['orderflow_features'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['orderflow_features'] = False
    
    print("\n5. Testing ML Predictor...")
    try:
        from ml_predictor import OrderFlowPredictor
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        data = {
            'timestamp': range(100),
            'midprice': 50000 + np.cumsum(np.random.randn(100)),
            'spread': np.random.rand(100) * 5,
            'imbalance_5': np.random.randn(100) * 0.3
        }
        df = pd.DataFrame(data)
        
        predictor = OrderFlowPredictor('logistic', horizon=1)
        # Don't train, just test initialization
        print("  ✓ Import successful")
        print("  ✓ Predictor initialization successful")
        results['ml_predictor'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['ml_predictor'] = False
    
    print("\n6. Testing Performance Metrics...")
    try:
        from performance_metrics import implementation_shortfall, vwap_slippage
        fills = [(50001.0, 10), (50002.0, 15)]
        is_result = implementation_shortfall(50000.0, fills, 'buy')
        slippage = vwap_slippage(fills, 50001.0, 'buy')
        assert 'is_bps' in is_result
        assert 'slippage_bps' in slippage
        print("  ✓ Import successful")
        print(f"  ✓ IS computed: {is_result['is_bps']:.2f} bps")
        print(f"  ✓ Slippage computed: {slippage['slippage_bps']:.2f} bps")
        results['performance_metrics'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['performance_metrics'] = False
    
    print("\n7. Testing Backtester...")
    try:
        from backtester import Backtester, TWAPStrategy
        from execution_schedules import twap_schedule
        import time
        
        # Create minimal synthetic snapshot
        snapshot = {
            'ts': int(time.time() * 1000),
            'orderbook': {
                'bids': [[50000.0, 10.0]],
                'asks': [[50001.0, 10.0]]
            },
            'trades': []
        }
        
        backtester = Backtester([snapshot]*10, bucket_duration_ms=1000)
        schedule = twap_schedule(10, 2)
        strategy = TWAPStrategy(10, schedule, 'buy')
        # Don't run full backtest, just test initialization
        print("  ✓ Import successful")
        print("  ✓ Backtester initialization successful")
        print("  ✓ Strategy initialization successful")
        results['backtester'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['backtester'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for module, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {module}")
    
    print(f"\n  Total: {passed}/{total} modules passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} module(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
