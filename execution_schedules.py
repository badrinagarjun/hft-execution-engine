"""
Execution schedule algorithms for algorithmic trading.
Implements TWAP, VWAP, and POV (Percentage of Volume) scheduling.
"""

import numpy as np
from typing import List, Dict, Optional
import pandas as pd


def twap_schedule(total_qty: float, periods: int) -> List[float]:
    """
    Time-Weighted Average Price (TWAP) schedule.
    Splits quantity equally across time periods.
    
    Args:
        total_qty: Total quantity to execute
        periods: Number of time periods
        
    Returns:
        List of quantities per period
    """
    base = total_qty // periods
    slices = [base] * periods
    
    # Distribute remainder
    remainder = int(total_qty - base * periods)
    for i in range(remainder):
        slices[i] += 1
    
    return [float(s) for s in slices]


def vwap_schedule(total_qty: float, volume_profile: List[float]) -> List[float]:
    """
    Volume-Weighted Average Price (VWAP) schedule.
    Allocates quantity proportional to expected volume in each period.
    
    Args:
        total_qty: Total quantity to execute
        volume_profile: Expected volume per period (e.g., historical intraday pattern)
        
    Returns:
        List of quantities per period
    """
    # Normalize volume profile to weights
    volume_array = np.array(volume_profile)
    weights = volume_array / np.sum(volume_array)
    
    # Calculate raw allocation
    raw_slices = weights * total_qty
    
    # Integer rounding with remainder distribution
    slices = np.floor(raw_slices).astype(int)
    remainder = int(total_qty - slices.sum())
    
    # Distribute remainder to periods with highest fractional parts
    fractional_parts = raw_slices - slices
    top_indices = np.argsort(-fractional_parts)
    
    for i in range(remainder):
        slices[top_indices[i % len(slices)]] += 1
    
    return slices.tolist()


def pov_schedule(total_qty: float, observed_volumes: List[float], 
                 participation_rate: float = 0.01, max_slice: Optional[float] = None) -> List[float]:
    """
    Percentage of Volume (POV) schedule.
    Allocates quantity as a percentage of observed market volume.
    
    Args:
        total_qty: Total quantity to execute
        observed_volumes: Observed market volumes per period
        participation_rate: Target participation rate (e.g., 0.01 = 1%)
        max_slice: Maximum slice size per period (None = no limit)
        
    Returns:
        List of quantities per period
    """
    slices = []
    remaining = total_qty
    
    for volume in observed_volumes:
        # Calculate slice as percentage of volume
        slice_size = participation_rate * volume
        
        # Apply max slice constraint
        if max_slice is not None:
            slice_size = min(slice_size, max_slice)
        
        # Don't exceed remaining quantity
        slice_size = min(slice_size, remaining)
        
        slices.append(slice_size)
        remaining -= slice_size
        
        if remaining <= 0:
            break
    
    # If there's remaining quantity after all periods, distribute it
    if remaining > 0 and slices:
        # Add remainder to last non-zero slices proportionally
        non_zero_indices = [i for i, s in enumerate(slices) if s > 0]
        if non_zero_indices:
            extra_per_slice = remaining / len(non_zero_indices)
            for idx in non_zero_indices:
                slices[idx] += extra_per_slice
    
    return slices


def adaptive_pov_schedule(total_qty: float, 
                          observed_volumes: List[float],
                          urgency_weights: List[float],
                          base_participation: float = 0.01,
                          max_participation: float = 0.05) -> List[float]:
    """
    Adaptive POV schedule that adjusts participation based on urgency.
    
    Args:
        total_qty: Total quantity to execute
        observed_volumes: Observed market volumes per period
        urgency_weights: Urgency weights per period (higher = more urgent)
        base_participation: Base participation rate
        max_participation: Maximum participation rate
        
    Returns:
        List of quantities per period
    """
    # Normalize urgency weights
    urgency_array = np.array(urgency_weights)
    normalized_urgency = urgency_array / np.max(urgency_array)
    
    # Calculate adaptive participation rates
    participation_rates = base_participation + (max_participation - base_participation) * normalized_urgency
    
    slices = []
    remaining = total_qty
    
    for volume, rate in zip(observed_volumes, participation_rates):
        slice_size = min(rate * volume, remaining)
        slices.append(slice_size)
        remaining -= slice_size
        
        if remaining <= 0:
            break
    
    # Distribute any remainder
    if remaining > 0 and slices:
        # Add to last slice
        slices[-1] += remaining
    
    return slices


def generate_intraday_volume_profile(periods: int = 390, pattern: str = 'u_shaped') -> List[float]:
    """
    Generate realistic intraday volume profile.
    
    Args:
        periods: Number of time periods (e.g., 390 minutes for US equity market)
        pattern: Volume pattern type ('u_shaped', 'flat', 'exponential_decay')
        
    Returns:
        List of relative volumes per period
    """
    if pattern == 'u_shaped':
        # High volume at open and close, lower in middle
        x = np.linspace(0, 1, periods)
        profile = 2.0 * (x**2 + (1-x)**2)  # U-shaped curve
        
    elif pattern == 'flat':
        # Uniform distribution
        profile = np.ones(periods)
        
    elif pattern == 'exponential_decay':
        # High volume at open, decaying throughout day
        x = np.linspace(0, 1, periods)
        profile = np.exp(-3 * x)
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Add some noise
    noise = np.random.normal(1.0, 0.1, periods)
    profile = profile * noise
    profile = np.maximum(profile, 0.1)  # Floor at 0.1
    
    return profile.tolist()


def compare_schedules(total_qty: float, periods: int, volume_profile: Optional[List[float]] = None):
    """
    Compare different execution schedules.
    
    Args:
        total_qty: Total quantity to execute
        periods: Number of periods
        volume_profile: Optional volume profile (generated if None)
    """
    if volume_profile is None:
        volume_profile = generate_intraday_volume_profile(periods, pattern='u_shaped')
    
    # Generate schedules
    twap = twap_schedule(total_qty, periods)
    vwap = vwap_schedule(total_qty, volume_profile)
    pov_1pct = pov_schedule(total_qty, volume_profile, participation_rate=0.01)
    pov_2pct = pov_schedule(total_qty, volume_profile, participation_rate=0.02)
    
    # Create comparison DataFrame
    df = pd.DataFrame({
        'Period': range(1, periods + 1),
        'Volume_Profile': volume_profile,
        'TWAP': twap + [0] * (periods - len(twap)),
        'VWAP': vwap + [0] * (periods - len(vwap)),
        'POV_1pct': pov_1pct + [0] * (periods - len(pov_1pct)),
        'POV_2pct': pov_2pct + [0] * (periods - len(pov_2pct))
    })
    
    return df


def test_schedules():
    """Test execution schedule algorithms."""
    print("=" * 60)
    print("Testing Execution Schedules")
    print("=" * 60)
    
    total_qty = 10000
    periods = 20
    
    # Test TWAP
    print(f"\n1. TWAP Schedule (Total: {total_qty}, Periods: {periods})")
    twap = twap_schedule(total_qty, periods)
    print(f"   Slices: {twap[:5]}... (first 5)")
    print(f"   Sum: {sum(twap)} (should equal {total_qty})")
    print(f"   Min: {min(twap)}, Max: {max(twap)}, Avg: {np.mean(twap):.2f}")
    
    # Test VWAP with U-shaped profile
    print(f"\n2. VWAP Schedule (U-shaped volume profile)")
    volume_profile = generate_intraday_volume_profile(periods, pattern='u_shaped')
    vwap = vwap_schedule(total_qty, volume_profile)
    print(f"   Slices: {[int(v) for v in vwap[:5]]}... (first 5)")
    print(f"   Sum: {sum(vwap)} (should equal {total_qty})")
    print(f"   Min: {min(vwap):.0f}, Max: {max(vwap):.0f}, Avg: {np.mean(vwap):.2f}")
    
    # Test POV
    print(f"\n3. POV Schedule (1% participation)")
    observed_volumes = [v * 1000 for v in volume_profile]  # Scale up
    pov = pov_schedule(total_qty, observed_volumes, participation_rate=0.01)
    print(f"   Slices: {[int(v) for v in pov[:5]]}... (first 5)")
    print(f"   Sum: {sum(pov):.0f} (target: {total_qty})")
    print(f"   Periods used: {len([p for p in pov if p > 0])}/{periods}")
    
    # Test Adaptive POV
    print(f"\n4. Adaptive POV Schedule (increasing urgency)")
    urgency = list(range(1, periods + 1))  # Linear increase
    adaptive = adaptive_pov_schedule(total_qty, observed_volumes, urgency, 
                                    base_participation=0.005, max_participation=0.03)
    print(f"   Slices: {[int(v) for v in adaptive[:5]]}... (first 5)")
    print(f"   Sum: {sum(adaptive):.0f} (target: {total_qty})")
    print(f"   Acceleration: {adaptive[-1]/adaptive[0]:.2f}x (last/first slice)")
    
    # Compare all schedules
    print(f"\n5. Schedule Comparison")
    comparison = compare_schedules(total_qty, periods)
    print("\n" + comparison.head(10).to_string(index=False))
    
    # Summary statistics
    print(f"\n6. Summary Statistics")
    print(f"   {'Strategy':<15} {'Mean':<10} {'Std':<10} {'Max/Min':<10}")
    print(f"   {'-'*45}")
    for col in ['TWAP', 'VWAP', 'POV_1pct', 'POV_2pct']:
        values = comparison[col][comparison[col] > 0]
        if len(values) > 0:
            print(f"   {col:<15} {values.mean():<10.2f} {values.std():<10.2f} {values.max()/values.min():<10.2f}")
    
    print("\n" + "=" * 60)
    print("Schedule tests complete!")


if __name__ == "__main__":
    test_schedules()
