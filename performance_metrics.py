"""
Performance metrics for execution strategies.
Implements Implementation Shortfall, VWAP slippage, and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy import stats


def implementation_shortfall(arrival_price: float, 
                            fills: List[Tuple[float, float]], 
                            side: str = 'buy') -> Dict:
    """
    Compute Implementation Shortfall (IS).
    
    IS measures the difference between the theoretical cost at decision time
    and the actual execution cost.
    
    Args:
        arrival_price: Price at decision/arrival time
        fills: List of (price, size) tuples
        side: 'buy' or 'sell'
        
    Returns:
        Dict with IS metrics (absolute, bps, percentage)
    """
    if not fills:
        return {'is_absolute': np.nan, 'is_bps': np.nan, 'is_pct': np.nan}
    
    # Calculate average execution price
    total_cost = sum(price * size for price, size in fills)
    total_size = sum(size for _, size in fills)
    avg_exec_price = total_cost / total_size if total_size > 0 else np.nan
    
    # Implementation shortfall
    if side == 'buy':
        is_absolute = avg_exec_price - arrival_price
    else:  # sell
        is_absolute = arrival_price - avg_exec_price
    
    # Convert to basis points and percentage
    is_bps = (is_absolute / arrival_price) * 10000 if arrival_price > 0 else np.nan
    is_pct = (is_absolute / arrival_price) * 100 if arrival_price > 0 else np.nan
    
    return {
        'arrival_price': arrival_price,
        'avg_exec_price': avg_exec_price,
        'is_absolute': is_absolute,
        'is_bps': is_bps,
        'is_pct': is_pct,
        'total_size': total_size,
        'num_fills': len(fills)
    }


def vwap_slippage(fills: List[Tuple[float, float]], 
                 benchmark_vwap: float,
                 side: str = 'buy') -> Dict:
    """
    Compute slippage relative to VWAP benchmark.
    
    Args:
        fills: List of (price, size) tuples
        benchmark_vwap: Benchmark VWAP price
        side: 'buy' or 'sell'
        
    Returns:
        Dict with slippage metrics
    """
    if not fills:
        return {'slippage_absolute': np.nan, 'slippage_bps': np.nan}
    
    # Calculate average execution price
    total_cost = sum(price * size for price, size in fills)
    total_size = sum(size for _, size in fills)
    avg_exec_price = total_cost / total_size if total_size > 0 else np.nan
    
    # Slippage
    if side == 'buy':
        slippage_absolute = avg_exec_price - benchmark_vwap
    else:  # sell
        slippage_absolute = benchmark_vwap - avg_exec_price
    
    slippage_bps = (slippage_absolute / benchmark_vwap) * 10000 if benchmark_vwap > 0 else np.nan
    
    return {
        'avg_exec_price': avg_exec_price,
        'benchmark_vwap': benchmark_vwap,
        'slippage_absolute': slippage_absolute,
        'slippage_bps': slippage_bps
    }


def fill_rate(scheduled_size: float, executed_size: float) -> float:
    """
    Compute fill rate (percentage of scheduled size executed).
    
    Args:
        scheduled_size: Total scheduled size
        executed_size: Total executed size
        
    Returns:
        Fill rate as percentage
    """
    if scheduled_size == 0:
        return 0.0
    return (executed_size / scheduled_size) * 100


def time_to_fill(fills: List[Dict], start_time: float, target_time: Optional[float] = None) -> Dict:
    """
    Compute time-to-fill metrics.
    
    Args:
        fills: List of fill dicts with 'timestamp' and 'size'
        start_time: Start timestamp (ms)
        target_time: Target completion time (ms), if any
        
    Returns:
        Dict with time metrics
    """
    if not fills:
        return {
            'total_time_ms': np.nan,
            'total_time_seconds': np.nan,
            'completed': False,
            'vs_target': np.nan
        }
    
    # Last fill time
    last_fill_time = max(fill['timestamp'] for fill in fills)
    total_time_ms = last_fill_time - start_time
    total_time_seconds = total_time_ms / 1000.0
    
    result = {
        'start_time': start_time,
        'end_time': last_fill_time,
        'total_time_ms': total_time_ms,
        'total_time_seconds': total_time_seconds,
        'completed': True,
        'num_fills': len(fills)
    }
    
    if target_time is not None:
        result['target_time'] = target_time
        result['vs_target_ms'] = last_fill_time - target_time
        result['on_time'] = last_fill_time <= target_time
    
    return result


def realized_spread(entry_price: float, exit_price: float, 
                   side: str, size: float, fees: float = 0.0) -> Dict:
    """
    Compute realized spread for market making.
    
    Args:
        entry_price: Entry execution price
        exit_price: Exit execution price
        side: Entry side ('buy' or 'sell')
        size: Position size
        fees: Total fees paid
        
    Returns:
        Dict with P&L metrics
    """
    if side == 'buy':
        pnl = (exit_price - entry_price) * size - fees
    else:  # sell
        pnl = (entry_price - exit_price) * size - fees
    
    pnl_bps = (pnl / (entry_price * size)) * 10000 if entry_price * size > 0 else np.nan
    
    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_bps': pnl_bps,
        'fees': fees,
        'net_pnl': pnl
    }


def market_impact(fills: List[Tuple[float, float]], 
                 midprice_before: float,
                 midprice_after: float,
                 side: str) -> Dict:
    """
    Compute market impact (temporary and permanent).
    
    Temporary impact: difference between execution price and arrival price
    Permanent impact: difference between arrival price and price after execution
    
    Args:
        fills: List of (price, size) tuples
        midprice_before: Midprice before execution
        midprice_after: Midprice after execution
        side: 'buy' or 'sell'
        
    Returns:
        Dict with impact metrics
    """
    if not fills:
        return {'temporary_impact': np.nan, 'permanent_impact': np.nan, 'total_impact': np.nan}
    
    # Average execution price
    total_cost = sum(price * size for price, size in fills)
    total_size = sum(size for _, size in fills)
    avg_price = total_cost / total_size if total_size > 0 else np.nan
    
    # Temporary impact (execution cost)
    if side == 'buy':
        temporary_impact = avg_price - midprice_before
        permanent_impact = midprice_after - midprice_before
    else:  # sell
        temporary_impact = midprice_before - avg_price
        permanent_impact = midprice_before - midprice_after
    
    total_impact = temporary_impact + permanent_impact
    
    # Convert to basis points
    temp_bps = (temporary_impact / midprice_before) * 10000 if midprice_before > 0 else np.nan
    perm_bps = (permanent_impact / midprice_before) * 10000 if midprice_before > 0 else np.nan
    total_bps = (total_impact / midprice_before) * 10000 if midprice_before > 0 else np.nan
    
    return {
        'midprice_before': midprice_before,
        'avg_exec_price': avg_price,
        'midprice_after': midprice_after,
        'temporary_impact': temporary_impact,
        'permanent_impact': permanent_impact,
        'total_impact': total_impact,
        'temporary_impact_bps': temp_bps,
        'permanent_impact_bps': perm_bps,
        'total_impact_bps': total_bps
    }


def paired_t_test(strategy_a_results: List[float], 
                 strategy_b_results: List[float]) -> Dict:
    """
    Perform paired t-test to compare two strategies.
    
    Args:
        strategy_a_results: List of metric values for strategy A
        strategy_b_results: List of metric values for strategy B
        
    Returns:
        Dict with test statistics
    """
    if len(strategy_a_results) != len(strategy_b_results):
        raise ValueError("Strategies must have same number of results")
    
    if len(strategy_a_results) < 2:
        return {
            'mean_diff': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False
        }
    
    # Compute differences
    differences = np.array(strategy_a_results) - np.array(strategy_b_results)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(strategy_a_results, strategy_b_results)
    
    return {
        'n_pairs': len(differences),
        'mean_a': np.mean(strategy_a_results),
        'mean_b': np.mean(strategy_b_results),
        'mean_diff': np.mean(differences),
        'std_diff': np.std(differences),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05,
        'significant_1pct': p_value < 0.01
    }


def bootstrap_confidence_interval(samples: List[float], 
                                  n_bootstrap: int = 10000,
                                  confidence: float = 0.95) -> Dict:
    """
    Compute bootstrap confidence interval.
    
    Args:
        samples: List of sample values
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Dict with confidence interval
    """
    if len(samples) < 2:
        return {
            'mean': np.nan,
            'lower': np.nan,
            'upper': np.nan
        }
    
    samples_array = np.array(samples)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(samples_array, size=len(samples_array), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return {
        'mean': np.mean(samples),
        'std': np.std(samples),
        'lower': lower,
        'upper': upper,
        'confidence': confidence,
        'n_bootstrap': n_bootstrap
    }


def compute_execution_summary(results: List[Dict]) -> pd.DataFrame:
    """
    Compute summary statistics for execution results.
    
    Args:
        results: List of result dicts from multiple executions
        
    Returns:
        DataFrame with summary statistics
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Compute summary
    summary = df.describe()
    
    # Add additional metrics
    if 'is_bps' in df.columns:
        summary.loc['median'] = df.median()
        summary.loc['skew'] = df.skew()
        summary.loc['kurtosis'] = df.kurtosis()
    
    return summary


def test_metrics():
    """Test performance metrics."""
    print("=" * 60)
    print("Testing Performance Metrics")
    print("=" * 60)
    
    # Test Implementation Shortfall
    print("\n1. Implementation Shortfall")
    arrival = 50000.0
    fills = [(50001.5, 10), (50002.0, 15), (50001.0, 5)]
    is_metrics = implementation_shortfall(arrival, fills, side='buy')
    print(f"   Arrival price: {is_metrics['arrival_price']:.2f}")
    print(f"   Avg exec price: {is_metrics['avg_exec_price']:.2f}")
    print(f"   IS (absolute): {is_metrics['is_absolute']:.2f}")
    print(f"   IS (bps): {is_metrics['is_bps']:.2f}")
    print(f"   IS (pct): {is_metrics['is_pct']:.4f}%")
    
    # Test VWAP slippage
    print("\n2. VWAP Slippage")
    benchmark_vwap = 50000.5
    slippage = vwap_slippage(fills, benchmark_vwap, side='buy')
    print(f"   Benchmark VWAP: {slippage['benchmark_vwap']:.2f}")
    print(f"   Avg exec price: {slippage['avg_exec_price']:.2f}")
    print(f"   Slippage (absolute): {slippage['slippage_absolute']:.2f}")
    print(f"   Slippage (bps): {slippage['slippage_bps']:.2f}")
    
    # Test fill rate
    print("\n3. Fill Rate")
    fr = fill_rate(scheduled_size=100, executed_size=95)
    print(f"   Fill rate: {fr:.2f}%")
    
    # Test market impact
    print("\n4. Market Impact")
    impact = market_impact(fills, midprice_before=50000.0, midprice_after=50002.5, side='buy')
    print(f"   Temporary impact: {impact['temporary_impact']:.2f} ({impact['temporary_impact_bps']:.2f} bps)")
    print(f"   Permanent impact: {impact['permanent_impact']:.2f} ({impact['permanent_impact_bps']:.2f} bps)")
    print(f"   Total impact: {impact['total_impact']:.2f} ({impact['total_impact_bps']:.2f} bps)")
    
    # Test paired t-test
    print("\n5. Paired T-Test (Strategy Comparison)")
    strategy_a = [1.5, 1.8, 1.2, 1.6, 1.9, 1.4, 1.7, 1.3, 1.8, 1.5]  # IS in bps
    strategy_b = [2.1, 2.3, 1.9, 2.0, 2.4, 2.2, 2.1, 1.8, 2.3, 2.0]  # IS in bps
    test_result = paired_t_test(strategy_a, strategy_b)
    print(f"   Strategy A mean: {test_result['mean_a']:.2f} bps")
    print(f"   Strategy B mean: {test_result['mean_b']:.2f} bps")
    print(f"   Mean difference: {test_result['mean_diff']:.2f} bps")
    print(f"   t-statistic: {test_result['t_statistic']:.4f}")
    print(f"   p-value: {test_result['p_value']:.6f}")
    print(f"   Significant at 5%: {test_result['significant_5pct']}")
    
    # Test bootstrap CI
    print("\n6. Bootstrap Confidence Interval")
    ci = bootstrap_confidence_interval(strategy_a, n_bootstrap=1000, confidence=0.95)
    print(f"   Mean: {ci['mean']:.2f} bps")
    print(f"   95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    
    # Test summary
    print("\n7. Execution Summary")
    results = [
        {'strategy': 'TWAP', 'is_bps': 1.5, 'fill_rate': 100.0},
        {'strategy': 'TWAP', 'is_bps': 1.8, 'fill_rate': 100.0},
        {'strategy': 'TWAP', 'is_bps': 1.2, 'fill_rate': 100.0},
        {'strategy': 'VWAP', 'is_bps': 1.1, 'fill_rate': 100.0},
        {'strategy': 'VWAP', 'is_bps': 1.3, 'fill_rate': 100.0},
        {'strategy': 'VWAP', 'is_bps': 0.9, 'fill_rate': 100.0},
    ]
    summary = compute_execution_summary(results)
    print(summary.to_string())
    
    print("\n" + "=" * 60)
    print("Metrics tests complete!")


if __name__ == "__main__":
    test_metrics()
