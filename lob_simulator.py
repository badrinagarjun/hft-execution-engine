"""
Simplified Limit Order Book (LOB) simulator for testing execution strategies.
Implements basic matching engine with limit orders, market orders, and cancellations.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class Order:
    """Represents a limit order in the book."""
    order_id: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    timestamp: float


class SimpleLOB:
    """
    Simplified Limit Order Book with matching engine.
    
    Features:
    - Add limit orders (buy/sell)
    - Cancel orders
    - Market orders (immediate execution)
    - Price-time priority matching
    """
    
    def __init__(self):
        """Initialize empty order book."""
        self.bids: List[List[float]] = []  # [[price, size], ...] sorted desc
        self.asks: List[List[float]] = []  # [[price, size], ...] sorted asc
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.next_order_id = 0
        
        # Statistics
        self.trade_history: List[Tuple[float, float, str]] = []  # (price, size, side)
        
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.next_order_id += 1
        return f"ORD_{self.next_order_id}"
    
    def add_limit(self, side: str, price: float, size: float, order_id: Optional[str] = None) -> str:
        """
        Add limit order to the book.
        
        Args:
            side: 'buy' or 'sell'
            price: Limit price
            size: Order size
            order_id: Optional order ID (auto-generated if None)
            
        Returns:
            Order ID
        """
        if order_id is None:
            order_id = self._generate_order_id()
        
        # Create order
        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            timestamp=time.time()
        )
        self.orders[order_id] = order
        
        # Add to appropriate book side
        if side == 'buy':
            self.bids.append([price, size, order_id])
            self.bids.sort(key=lambda x: -x[0])  # Sort descending by price
        else:
            self.asks.append([price, size, order_id])
            self.asks.sort(key=lambda x: x[0])  # Sort ascending by price
        
        return order_id
    
    def cancel(self, order_id: str) -> bool:
        """
        Cancel order by ID.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Remove from appropriate book side
        if order.side == 'buy':
            self.bids = [[p, s, oid] for p, s, oid in self.bids if oid != order_id]
        else:
            self.asks = [[p, s, oid] for p, s, oid in self.asks if oid != order_id]
        
        del self.orders[order_id]
        return True
    
    def market_buy(self, size: float) -> List[Tuple[float, float]]:
        """
        Execute market buy order (take liquidity from asks).
        
        Args:
            size: Size to buy
            
        Returns:
            List of (price, executed_size) tuples
        """
        executed = []
        remaining = size
        
        while remaining > 0 and self.asks:
            best_ask = self.asks[0]
            price, available, order_id = best_ask[0], best_ask[1], best_ask[2]
            
            # Determine how much to take
            take = min(available, remaining)
            executed.append((price, take))
            remaining -= take
            
            # Update or remove from book
            if take >= available:
                self.asks.pop(0)
                if order_id in self.orders:
                    del self.orders[order_id]
            else:
                self.asks[0][1] -= take
                if order_id in self.orders:
                    self.orders[order_id].size -= take
            
            # Record trade
            self.trade_history.append((price, take, 'buy'))
        
        return executed
    
    def market_sell(self, size: float) -> List[Tuple[float, float]]:
        """
        Execute market sell order (take liquidity from bids).
        
        Args:
            size: Size to sell
            
        Returns:
            List of (price, executed_size) tuples
        """
        executed = []
        remaining = size
        
        while remaining > 0 and self.bids:
            best_bid = self.bids[0]
            price, available, order_id = best_bid[0], best_bid[1], best_bid[2]
            
            # Determine how much to take
            take = min(available, remaining)
            executed.append((price, take))
            remaining -= take
            
            # Update or remove from book
            if take >= available:
                self.bids.pop(0)
                if order_id in self.orders:
                    del self.orders[order_id]
            else:
                self.bids[0][1] -= take
                if order_id in self.orders:
                    self.orders[order_id].size -= take
            
            # Record trade
            self.trade_history.append((price, take, 'sell'))
        
        return executed
    
    def get_midprice(self) -> Optional[float]:
        """Get current midprice."""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    def get_spread(self) -> Optional[float]:
        """Get current spread."""
        if not self.bids or not self.asks:
            return None
        return self.asks[0][0] - self.bids[0][0]
    
    def get_depth(self, levels: int = 5) -> Dict:
        """
        Get order book depth.
        
        Args:
            levels: Number of levels to return
            
        Returns:
            Dict with bids and asks
        """
        return {
            'bids': [[p, s] for p, s, _ in self.bids[:levels]],
            'asks': [[p, s] for p, s, _ in self.asks[:levels]],
            'midprice': self.get_midprice(),
            'spread': self.get_spread()
        }
    
    def update_from_snapshot(self, snapshot: Dict):
        """
        Update LOB from orderbook snapshot (e.g., from data_capture).
        
        Args:
            snapshot: Snapshot dict with 'orderbook' containing 'bids' and 'asks'
        """
        # Clear existing book
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        
        # Load snapshot
        ob = snapshot.get('orderbook', snapshot)
        
        # Add bids
        for price, size in ob.get('bids', []):
            order_id = self._generate_order_id()
            self.bids.append([price, size, order_id])
        
        # Add asks
        for price, size in ob.get('asks', []):
            order_id = self._generate_order_id()
            self.asks.append([price, size, order_id])
        
        # Already sorted from exchange data
        self.bids.sort(key=lambda x: -x[0])
        self.asks.sort(key=lambda x: x[0])
    
    def __repr__(self) -> str:
        """String representation of book state."""
        depth = self.get_depth(levels=5)
        lines = ["SimpleLOB State:"]
        lines.append(f"  Midprice: {depth['midprice']:.2f}" if depth['midprice'] else "  Midprice: N/A")
        lines.append(f"  Spread: {depth['spread']:.4f}" if depth['spread'] else "  Spread: N/A")
        lines.append("  Asks:")
        for price, size in reversed(depth['asks']):
            lines.append(f"    {price:>10.2f} | {size:>8.4f}")
        lines.append("  " + "-" * 25)
        lines.append("  Bids:")
        for price, size in depth['bids']:
            lines.append(f"    {price:>10.2f} | {size:>8.4f}")
        return "\n".join(lines)


def test_lob():
    """Test LOB simulator functionality."""
    print("=" * 60)
    print("Testing SimpleLOB Simulator")
    print("=" * 60)
    
    lob = SimpleLOB()
    
    # Add some limit orders
    print("\n1. Adding limit orders...")
    lob.add_limit('buy', 100.00, 10.0)
    lob.add_limit('buy', 99.50, 5.0)
    lob.add_limit('buy', 99.00, 8.0)
    lob.add_limit('sell', 100.50, 7.0)
    lob.add_limit('sell', 101.00, 12.0)
    lob.add_limit('sell', 101.50, 6.0)
    
    print(lob)
    
    # Test market buy
    print("\n2. Executing market buy of 10.0...")
    fills = lob.market_buy(10.0)
    print(f"Fills: {fills}")
    print(f"Average price: {sum(p*s for p,s in fills) / sum(s for _,s in fills):.2f}")
    print(lob)
    
    # Test market sell
    print("\n3. Executing market sell of 8.0...")
    fills = lob.market_sell(8.0)
    print(f"Fills: {fills}")
    print(f"Average price: {sum(p*s for p,s in fills) / sum(s for _,s in fills):.2f}")
    print(lob)
    
    # Test cancellation
    print("\n4. Testing order cancellation...")
    order_id = lob.add_limit('buy', 98.50, 15.0)
    print(f"Added order {order_id}")
    print(lob)
    
    print(f"\nCancelling order {order_id}...")
    lob.cancel(order_id)
    print(lob)
    
    # Test with real snapshot data
    print("\n5. Testing with snapshot data...")
    mock_snapshot = {
        'orderbook': {
            'bids': [[50000.0, 1.5], [49999.0, 2.0], [49998.0, 0.8]],
            'asks': [[50001.0, 1.2], [50002.0, 1.8], [50003.0, 2.5]]
        }
    }
    
    lob.update_from_snapshot(mock_snapshot)
    print(lob)
    
    print("\n" + "=" * 60)
    print("LOB tests complete!")


if __name__ == "__main__":
    test_lob()
