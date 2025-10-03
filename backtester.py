"""
Event-driven backtester for execution strategies.
Replays historical market data and simulates order execution.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time

from lob_simulator import SimpleLOB
from orderflow_features import extract_features_from_snapshot
from performance_metrics import implementation_shortfall, fill_rate, market_impact


class EventType(Enum):
    """Types of events in the backtester."""
    MARKET_UPDATE = "market_update"
    TIME_BUCKET = "time_bucket"
    ORDER_FILL = "order_fill"
    SCHEDULE_COMPLETE = "schedule_complete"


@dataclass
class Event:
    """Represents an event in the backtest."""
    timestamp: float
    event_type: EventType
    data: Dict


class ExecutionStrategy:
    """Base class for execution strategies."""
    
    def __init__(self, total_qty: float, schedule: List[float]):
        """
        Initialize strategy.
        
        Args:
            total_qty: Total quantity to execute
            schedule: List of quantities per time bucket
        """
        self.total_qty = total_qty
        self.schedule = schedule
        self.current_bucket = 0
        self.fills = []
        self.arrival_price = None
        
    def on_market_update(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        """
        Handle market update event.
        
        Args:
            lob: Current LOB state
            features: Current orderflow features
            
        Returns:
            List of orders to submit [{'type': 'market'/'limit', 'side': 'buy'/'sell', 'price': float, 'size': float}]
        """
        raise NotImplementedError
    
    def on_time_bucket(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        """
        Handle time bucket event (scheduled execution time).
        
        Args:
            lob: Current LOB state
            features: Current orderflow features
            
        Returns:
            List of orders to submit
        """
        raise NotImplementedError
    
    def record_fill(self, price: float, size: float, timestamp: float):
        """Record a fill."""
        self.fills.append({'price': price, 'size': size, 'timestamp': timestamp})
        
        # Set arrival price on first fill
        if self.arrival_price is None:
            self.arrival_price = price


class TWAPStrategy(ExecutionStrategy):
    """TWAP execution strategy (aggressive market orders)."""
    
    def __init__(self, total_qty: float, schedule: List[float], side: str = 'buy'):
        super().__init__(total_qty, schedule)
        self.side = side
        
    def on_market_update(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        return []  # Only act on time buckets
    
    def on_time_bucket(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        if self.current_bucket >= len(self.schedule):
            return []
        
        qty = self.schedule[self.current_bucket]
        self.current_bucket += 1
        
        if qty > 0:
            return [{'type': 'market', 'side': self.side, 'size': qty}]
        return []


class VWAPStrategy(ExecutionStrategy):
    """VWAP execution strategy (aggressive market orders on schedule)."""
    
    def __init__(self, total_qty: float, schedule: List[float], side: str = 'buy'):
        super().__init__(total_qty, schedule)
        self.side = side
        
    def on_market_update(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        return []  # Only act on time buckets
    
    def on_time_bucket(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        if self.current_bucket >= len(self.schedule):
            return []
        
        qty = self.schedule[self.current_bucket]
        self.current_bucket += 1
        
        if qty > 0:
            return [{'type': 'market', 'side': self.side, 'size': qty}]
        return []


class AdaptiveStrategy(ExecutionStrategy):
    """Adaptive execution strategy using ML predictions."""
    
    def __init__(self, total_qty: float, schedule: List[float], side: str = 'buy',
                 predictor: Optional[Callable] = None, aggressive_threshold: float = 0.65,
                 passive_threshold: float = 0.35):
        super().__init__(total_qty, schedule)
        self.side = side
        self.predictor = predictor  # Function that takes features and returns probability
        self.aggressive_threshold = aggressive_threshold
        self.passive_threshold = passive_threshold
        self.pending_limit_orders = []
        
    def on_market_update(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        # Check and cancel stale limit orders if needed
        # For simplicity, just clear them
        self.pending_limit_orders = []
        return []
    
    def on_time_bucket(self, lob: SimpleLOB, features: Dict) -> List[Dict]:
        if self.current_bucket >= len(self.schedule):
            return []
        
        qty = self.schedule[self.current_bucket]
        self.current_bucket += 1
        
        if qty <= 0:
            return []
        
        # If no predictor, default to market orders
        if self.predictor is None:
            return [{'type': 'market', 'side': self.side, 'size': qty}]
        
        # Get prediction
        try:
            prediction_proba = self.predictor(features)
        except:
            # Fallback to market order if prediction fails
            return [{'type': 'market', 'side': self.side, 'size': qty}]
        
        # Decision logic
        if self.side == 'buy':
            # If predicting price will go up, be aggressive (buy now)
            if prediction_proba > self.aggressive_threshold:
                return [{'type': 'market', 'side': 'buy', 'size': qty}]
            # If predicting price will go down, be passive (wait or use limit)
            elif prediction_proba < self.passive_threshold:
                # Post limit order at best bid (or skip this slice)
                if lob.bids:
                    limit_price = lob.bids[0][0]
                    return [{'type': 'limit', 'side': 'buy', 'price': limit_price, 'size': qty}]
                else:
                    return [{'type': 'market', 'side': 'buy', 'size': qty}]
            # Neutral: follow schedule
            else:
                return [{'type': 'market', 'side': 'buy', 'size': qty}]
        else:  # sell side
            # If predicting price will go down, be aggressive (sell now)
            if prediction_proba < self.passive_threshold:
                return [{'type': 'market', 'side': 'sell', 'size': qty}]
            # If predicting price will go up, be passive
            elif prediction_proba > self.aggressive_threshold:
                if lob.asks:
                    limit_price = lob.asks[0][0]
                    return [{'type': 'limit', 'side': 'sell', 'price': limit_price, 'size': qty}]
                else:
                    return [{'type': 'market', 'side': 'sell', 'size': qty}]
            else:
                return [{'type': 'market', 'side': 'sell', 'size': qty}]


class Backtester:
    """Event-driven backtester for execution strategies."""
    
    def __init__(self, snapshots: List[Dict], bucket_duration_ms: int = 1000):
        """
        Initialize backtester.
        
        Args:
            snapshots: List of market snapshots
            bucket_duration_ms: Duration of each time bucket in milliseconds
        """
        self.snapshots = sorted(snapshots, key=lambda x: x['ts'])
        self.bucket_duration_ms = bucket_duration_ms
        self.lob = SimpleLOB()
        self.results = []
        
    def run(self, strategy: ExecutionStrategy) -> Dict:
        """
        Run backtest for given strategy.
        
        Args:
            strategy: Execution strategy to test
            
        Returns:
            Dict with backtest results
        """
        if not self.snapshots:
            return {'error': 'No snapshots to process'}
        
        # Initialize
        start_time = self.snapshots[0]['ts']
        end_time = self.snapshots[-1]['ts']
        current_bucket_start = start_time
        next_bucket_time = start_time + self.bucket_duration_ms
        
        # Track metrics
        arrival_mid = None
        final_mid = None
        bucket_count = 0
        
        # Process snapshots
        prev_snapshots = []
        
        for i, snapshot in enumerate(self.snapshots):
            current_time = snapshot['ts']
            
            # Update LOB
            self.lob.update_from_snapshot(snapshot)
            
            # Set arrival price on first snapshot
            if arrival_mid is None:
                arrival_mid = self.lob.get_midprice()
            
            # Extract features
            features = extract_features_from_snapshot(snapshot, prev_snapshots[-10:] if prev_snapshots else [])
            prev_snapshots.append(snapshot)
            
            # Check if we've crossed into a new time bucket
            if current_time >= next_bucket_time:
                # Time bucket event
                orders = strategy.on_time_bucket(self.lob, features)
                self._execute_orders(orders, current_time, strategy)
                
                next_bucket_time += self.bucket_duration_ms
                bucket_count += 1
            
            # Market update event
            orders = strategy.on_market_update(self.lob, features)
            self._execute_orders(orders, current_time, strategy)
        
        # Final metrics
        final_mid = self.lob.get_midprice()
        
        # Compute performance metrics
        if strategy.fills and arrival_mid:
            fills_tuples = [(f['price'], f['size']) for f in strategy.fills]
            
            is_metrics = implementation_shortfall(
                arrival_price=arrival_mid,
                fills=fills_tuples,
                side=getattr(strategy, 'side', 'buy')
            )
            
            impact = market_impact(
                fills=fills_tuples,
                midprice_before=arrival_mid,
                midprice_after=final_mid if final_mid else arrival_mid,
                side=getattr(strategy, 'side', 'buy')
            )
            
            total_executed = sum(f['size'] for f in strategy.fills)
            fr = fill_rate(strategy.total_qty, total_executed)
            
            return {
                'strategy': strategy.__class__.__name__,
                'start_time': start_time,
                'end_time': end_time,
                'duration_ms': end_time - start_time,
                'buckets': bucket_count,
                'arrival_mid': arrival_mid,
                'final_mid': final_mid,
                'total_qty': strategy.total_qty,
                'executed_qty': total_executed,
                'fill_rate': fr,
                'num_fills': len(strategy.fills),
                **is_metrics,
                **{f'impact_{k}': v for k, v in impact.items()}
            }
        else:
            return {
                'strategy': strategy.__class__.__name__,
                'error': 'No fills executed',
                'total_qty': strategy.total_qty,
                'executed_qty': 0
            }
    
    def _execute_orders(self, orders: List[Dict], timestamp: float, strategy: ExecutionStrategy):
        """Execute list of orders against current LOB."""
        for order in orders:
            order_type = order.get('type', 'market')
            side = order.get('side', 'buy')
            size = order.get('size', 0)
            
            if size <= 0:
                continue
            
            if order_type == 'market':
                # Execute market order
                if side == 'buy':
                    fills = self.lob.market_buy(size)
                else:
                    fills = self.lob.market_sell(size)
                
                # Record fills
                for price, filled_size in fills:
                    strategy.record_fill(price, filled_size, timestamp)
            
            elif order_type == 'limit':
                # Add limit order (simplified: immediately try to match)
                price = order.get('price')
                order_id = self.lob.add_limit(side, price, size)
                # In real implementation, would track pending limit orders
                # For simplicity, we'll assume they don't fill immediately


def test_backtester():
    """Test backtester with synthetic data."""
    print("=" * 60)
    print("Testing Backtester")
    print("=" * 60)
    
    # Generate synthetic market data
    print("\n1. Generating synthetic market data...")
    base_time = int(time.time() * 1000)
    base_price = 50000.0
    num_snapshots = 100
    
    snapshots = []
    for i in range(num_snapshots):
        # Random walk price
        price = base_price + np.cumsum(np.random.randn(i+1))[-1] * 10
        
        snapshot = {
            'ts': base_time + i * 1000,  # 1 second apart
            'orderbook': {
                'bids': [[price - j - 0.5, 1.0 + np.random.rand()] for j in range(10)],
                'asks': [[price + j + 0.5, 1.0 + np.random.rand()] for j in range(10)]
            },
            'trades': []
        }
        snapshots.append(snapshot)
    
    print(f"   Generated {len(snapshots)} snapshots")
    
    # Create backtester
    backtester = Backtester(snapshots, bucket_duration_ms=10000)  # 10 second buckets
    
    # Test TWAP strategy
    print("\n2. Testing TWAP Strategy...")
    from execution_schedules import twap_schedule
    total_qty = 100.0
    num_buckets = 10
    twap_sched = twap_schedule(total_qty, num_buckets)
    
    twap_strategy = TWAPStrategy(total_qty, twap_sched, side='buy')
    twap_results = backtester.run(twap_strategy)
    
    print(f"   Strategy: {twap_results.get('strategy')}")
    print(f"   Executed: {twap_results.get('executed_qty', 0):.2f} / {twap_results.get('total_qty', 0):.2f}")
    print(f"   Fill rate: {twap_results.get('fill_rate', 0):.2f}%")
    print(f"   Arrival mid: {twap_results.get('arrival_mid', 0):.2f}")
    print(f"   Avg exec price: {twap_results.get('avg_exec_price', 0):.2f}")
    print(f"   IS (bps): {twap_results.get('is_bps', 0):.2f}")
    
    # Test VWAP strategy
    print("\n3. Testing VWAP Strategy...")
    from execution_schedules import vwap_schedule, generate_intraday_volume_profile
    volume_profile = generate_intraday_volume_profile(num_buckets, pattern='u_shaped')
    vwap_sched = vwap_schedule(total_qty, volume_profile)
    
    # Reset backtester LOB
    backtester = Backtester(snapshots, bucket_duration_ms=10000)
    
    vwap_strategy = VWAPStrategy(total_qty, vwap_sched, side='buy')
    vwap_results = backtester.run(vwap_strategy)
    
    print(f"   Strategy: {vwap_results.get('strategy')}")
    print(f"   Executed: {vwap_results.get('executed_qty', 0):.2f} / {vwap_results.get('total_qty', 0):.2f}")
    print(f"   Fill rate: {vwap_results.get('fill_rate', 0):.2f}%")
    print(f"   IS (bps): {vwap_results.get('is_bps', 0):.2f}")
    
    # Compare strategies
    print("\n4. Strategy Comparison")
    print(f"   {'Metric':<20} {'TWAP':<15} {'VWAP':<15}")
    print(f"   {'-'*50}")
    print(f"   {'IS (bps)':<20} {twap_results.get('is_bps', 0):>14.2f} {vwap_results.get('is_bps', 0):>14.2f}")
    print(f"   {'Avg Price':<20} {twap_results.get('avg_exec_price', 0):>14.2f} {vwap_results.get('avg_exec_price', 0):>14.2f}")
    print(f"   {'Num Fills':<20} {twap_results.get('num_fills', 0):>14} {vwap_results.get('num_fills', 0):>14}")
    
    print("\n" + "=" * 60)
    print("Backtester tests complete!")
    
    return backtester, twap_results, vwap_results


if __name__ == "__main__":
    test_backtester()
