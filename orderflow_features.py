"""
Order-flow feature extraction for microstructure analysis.
Computes features from LOB snapshots for predictive modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


def compute_imbalance(bids: List[List[float]], asks: List[List[float]], 
                     depth: int = 5) -> float:
    """
    Compute order book imbalance at specified depth.
    
    Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    Args:
        bids: List of [price, size] for bids
        asks: List of [price, size] for asks
        depth: Number of levels to consider
        
    Returns:
        Imbalance value in [-1, 1]
    """
    bid_vol = sum([size for price, size in bids[:depth]])
    ask_vol = sum([size for price, size in asks[:depth]])
    
    if bid_vol + ask_vol == 0:
        return 0.0
    
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12)


def compute_spread(bids: List[List[float]], asks: List[List[float]]) -> float:
    """
    Compute bid-ask spread.
    
    Args:
        bids: List of [price, size] for bids
        asks: List of [price, size] for asks
        
    Returns:
        Spread (best_ask - best_bid)
    """
    if not bids or not asks:
        return np.nan
    
    return asks[0][0] - bids[0][0]


def compute_midprice(bids: List[List[float]], asks: List[List[float]]) -> float:
    """
    Compute midprice.
    
    Args:
        bids: List of [price, size] for bids
        asks: List of [price, size] for asks
        
    Returns:
        Midprice (best_bid + best_ask) / 2
    """
    if not bids or not asks:
        return np.nan
    
    return (bids[0][0] + asks[0][0]) / 2


def compute_weighted_midprice(bids: List[List[float]], asks: List[List[float]]) -> float:
    """
    Compute size-weighted midprice.
    
    Args:
        bids: List of [price, size] for bids
        asks: List of [price, size] for asks
        
    Returns:
        Weighted midprice
    """
    if not bids or not asks:
        return np.nan
    
    best_bid_price, best_bid_size = bids[0][0], bids[0][1]
    best_ask_price, best_ask_size = asks[0][0], asks[0][1]
    
    total_size = best_bid_size + best_ask_size
    if total_size == 0:
        return (best_bid_price + best_ask_price) / 2
    
    return (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / total_size


def compute_depth_slope(bids: List[List[float]], asks: List[List[float]], 
                       depth: int = 10) -> Tuple[float, float]:
    """
    Compute slope of cumulative depth (liquidity profile).
    
    Args:
        bids: List of [price, size] for bids
        asks: List of [price, size] for asks
        depth: Number of levels to consider
        
    Returns:
        (bid_slope, ask_slope) - steeper slope means faster depth accumulation
    """
    # Bid side
    bid_prices = np.array([p for p, s in bids[:depth]])
    bid_cumvol = np.cumsum([s for p, s in bids[:depth]])
    
    if len(bid_prices) > 1:
        bid_slope = np.polyfit(bid_prices, bid_cumvol, 1)[0]
    else:
        bid_slope = 0.0
    
    # Ask side
    ask_prices = np.array([p for p, s in asks[:depth]])
    ask_cumvol = np.cumsum([s for p, s in asks[:depth]])
    
    if len(ask_prices) > 1:
        ask_slope = np.polyfit(ask_prices, ask_cumvol, 1)[0]
    else:
        ask_slope = 0.0
    
    return bid_slope, ask_slope


def compute_trade_imbalance(trades: List[Dict], window_seconds: float = 1.0) -> float:
    """
    Compute signed trade volume imbalance over recent window.
    
    Args:
        trades: List of trade dicts with 'timestamp', 'price', 'amount', 'side'
        window_seconds: Time window in seconds
        
    Returns:
        Trade imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    """
    if not trades:
        return 0.0
    
    # Get recent trades
    latest_time = trades[-1].get('timestamp', 0)
    cutoff_time = latest_time - window_seconds * 1000  # Convert to ms
    
    buy_volume = 0.0
    sell_volume = 0.0
    
    for trade in trades:
        if trade.get('timestamp', 0) < cutoff_time:
            continue
        
        amount = trade.get('amount', 0)
        side = trade.get('side', 'unknown')
        
        if side == 'buy':
            buy_volume += amount
        elif side == 'sell':
            sell_volume += amount
        else:
            # If side unknown, use price vs midprice as proxy
            # This is a heuristic when side info is unavailable
            pass
    
    total = buy_volume + sell_volume
    if total == 0:
        return 0.0
    
    return (buy_volume - sell_volume) / (total + 1e-12)


def compute_price_impact(bids: List[List[float]], asks: List[List[float]], 
                        volume: float, side: str) -> float:
    """
    Estimate price impact of market order of given size.
    
    Args:
        bids: List of [price, size] for bids
        asks: List of [price, size] for asks
        volume: Order size
        side: 'buy' or 'sell'
        
    Returns:
        Estimated price impact in price units
    """
    levels = asks if side == 'buy' else bids
    
    if not levels:
        return np.nan
    
    initial_price = levels[0][0]
    remaining = volume
    total_cost = 0.0
    
    for price, size in levels:
        take = min(size, remaining)
        total_cost += price * take
        remaining -= take
        
        if remaining <= 0:
            break
    
    if remaining > 0:
        # Not enough liquidity
        return np.nan
    
    avg_price = total_cost / volume
    return abs(avg_price - initial_price)


def extract_features_from_snapshot(snapshot: Dict, 
                                   prev_snapshots: Optional[List[Dict]] = None) -> Dict:
    """
    Extract all features from a single snapshot.
    
    Args:
        snapshot: Current snapshot dict
        prev_snapshots: List of previous snapshots for time-series features
        
    Returns:
        Dict of features
    """
    ob = snapshot.get('orderbook', snapshot)
    bids = ob.get('bids', [])
    asks = ob.get('asks', [])
    trades = snapshot.get('trades', [])
    
    features = {
        'timestamp': snapshot.get('ts', 0),
        
        # Basic price features
        'midprice': compute_midprice(bids, asks),
        'spread': compute_spread(bids, asks),
        'weighted_midprice': compute_weighted_midprice(bids, asks),
        
        # Imbalance features (multiple depths)
        'imbalance_1': compute_imbalance(bids, asks, depth=1),
        'imbalance_5': compute_imbalance(bids, asks, depth=5),
        'imbalance_10': compute_imbalance(bids, asks, depth=10),
        
        # Depth features
        'bid_depth_5': sum([s for p, s in bids[:5]]),
        'ask_depth_5': sum([s for p, s in asks[:5]]),
        'bid_depth_10': sum([s for p, s in bids[:10]]),
        'ask_depth_10': sum([s for p, s in asks[:10]]),
        
        # Trade features
        'trade_imbalance': compute_trade_imbalance(trades, window_seconds=1.0),
        'num_trades': len(trades),
        
        # Price impact estimates
        'impact_buy_small': compute_price_impact(bids, asks, volume=0.1, side='buy'),
        'impact_sell_small': compute_price_impact(bids, asks, volume=0.1, side='sell'),
    }
    
    # Depth slopes
    bid_slope, ask_slope = compute_depth_slope(bids, asks, depth=10)
    features['bid_depth_slope'] = bid_slope
    features['ask_depth_slope'] = ask_slope
    
    # Time-series features (if previous snapshots available)
    if prev_snapshots and len(prev_snapshots) > 0:
        prev_mid = compute_midprice(
            prev_snapshots[-1].get('orderbook', {}).get('bids', []),
            prev_snapshots[-1].get('orderbook', {}).get('asks', [])
        )
        if not np.isnan(prev_mid) and not np.isnan(features['midprice']):
            features['midprice_change'] = features['midprice'] - prev_mid
            features['midprice_return'] = (features['midprice'] - prev_mid) / prev_mid
        else:
            features['midprice_change'] = 0.0
            features['midprice_return'] = 0.0
        
        # Spread change
        prev_spread = compute_spread(
            prev_snapshots[-1].get('orderbook', {}).get('bids', []),
            prev_snapshots[-1].get('orderbook', {}).get('asks', [])
        )
        features['spread_change'] = features['spread'] - prev_spread if not np.isnan(prev_spread) else 0.0
    else:
        features['midprice_change'] = 0.0
        features['midprice_return'] = 0.0
        features['spread_change'] = 0.0
    
    return features


def create_feature_matrix(snapshots: List[Dict], lookback: int = 10) -> pd.DataFrame:
    """
    Create feature matrix from list of snapshots.
    
    Args:
        snapshots: List of snapshot dicts
        lookback: Number of previous snapshots to use for time-series features
        
    Returns:
        DataFrame with features
    """
    features_list = []
    
    for i, snapshot in enumerate(snapshots):
        # Get previous snapshots for time-series features
        prev_snapshots = snapshots[max(0, i-lookback):i] if i > 0 else []
        
        features = extract_features_from_snapshot(snapshot, prev_snapshots)
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    return df


def create_labels(feature_df: pd.DataFrame, horizon: int = 1, 
                 threshold: float = 0.0) -> pd.Series:
    """
    Create labels for prediction (next tick direction).
    
    Args:
        feature_df: DataFrame with features including 'midprice'
        horizon: Prediction horizon (number of ticks ahead)
        threshold: Minimum price change to consider as signal
        
    Returns:
        Series of labels (1 = up, -1 = down, 0 = neutral)
    """
    future_midprice = feature_df['midprice'].shift(-horizon)
    current_midprice = feature_df['midprice']
    
    price_change = future_midprice - current_midprice
    
    labels = pd.Series(0, index=feature_df.index)
    labels[price_change > threshold] = 1
    labels[price_change < -threshold] = -1
    
    return labels


def test_features():
    """Test feature extraction."""
    print("=" * 60)
    print("Testing Order-Flow Feature Extraction")
    print("=" * 60)
    
    # Mock snapshot
    mock_snapshot = {
        'ts': int(time.time() * 1000),
        'orderbook': {
            'bids': [
                [50000.0, 1.5], [49999.0, 2.0], [49998.0, 0.8],
                [49997.0, 1.2], [49996.0, 2.5], [49995.0, 1.8]
            ],
            'asks': [
                [50001.0, 1.2], [50002.0, 1.8], [50003.0, 2.5],
                [50004.0, 0.9], [50005.0, 1.5], [50006.0, 2.1]
            ]
        },
        'trades': [
            {'timestamp': int(time.time() * 1000), 'price': 50000.5, 'amount': 0.5, 'side': 'buy'},
            {'timestamp': int(time.time() * 1000), 'price': 50001.0, 'amount': 0.3, 'side': 'sell'},
        ]
    }
    
    print("\n1. Testing individual feature functions...")
    bids = mock_snapshot['orderbook']['bids']
    asks = mock_snapshot['orderbook']['asks']
    
    print(f"   Midprice: {compute_midprice(bids, asks):.2f}")
    print(f"   Spread: {compute_spread(bids, asks):.2f}")
    print(f"   Imbalance (depth=5): {compute_imbalance(bids, asks, depth=5):.4f}")
    print(f"   Weighted midprice: {compute_weighted_midprice(bids, asks):.2f}")
    
    bid_slope, ask_slope = compute_depth_slope(bids, asks, depth=5)
    print(f"   Bid depth slope: {bid_slope:.4f}")
    print(f"   Ask depth slope: {ask_slope:.4f}")
    
    print(f"   Trade imbalance: {compute_trade_imbalance(mock_snapshot['trades']):.4f}")
    print(f"   Price impact (buy 0.1): {compute_price_impact(bids, asks, 0.1, 'buy'):.2f}")
    
    print("\n2. Testing full feature extraction...")
    features = extract_features_from_snapshot(mock_snapshot)
    print(f"   Total features: {len(features)}")
    print("   Sample features:")
    for key in ['midprice', 'spread', 'imbalance_5', 'bid_depth_5', 'trade_imbalance']:
        print(f"     {key}: {features.get(key, 'N/A')}")
    
    print("\n3. Testing feature matrix creation...")
    # Create multiple mock snapshots
    import time
    snapshots = []
    base_price = 50000.0
    for i in range(20):
        price = base_price + np.random.randn() * 10
        snapshot = {
            'ts': int(time.time() * 1000) + i * 1000,
            'orderbook': {
                'bids': [[price - j, 1.0 + np.random.rand()] for j in range(1, 11)],
                'asks': [[price + j, 1.0 + np.random.rand()] for j in range(1, 11)]
            },
            'trades': []
        }
        snapshots.append(snapshot)
    
    feature_df = create_feature_matrix(snapshots, lookback=5)
    print(f"   Feature matrix shape: {feature_df.shape}")
    print(f"   Columns: {list(feature_df.columns)}")
    print("\n   First few rows:")
    print(feature_df[['timestamp', 'midprice', 'spread', 'imbalance_5']].head())
    
    print("\n4. Testing label creation...")
    labels = create_labels(feature_df, horizon=1, threshold=0.5)
    print(f"   Label distribution:")
    print(f"     Up (1): {(labels == 1).sum()}")
    print(f"     Down (-1): {(labels == -1).sum()}")
    print(f"     Neutral (0): {(labels == 0).sum()}")
    
    print("\n" + "=" * 60)
    print("Feature extraction tests complete!")


if __name__ == "__main__":
    import time
    test_features()
