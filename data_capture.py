"""
Data capture module for live/historical microdata from crypto exchanges.
Uses ccxt to pull order book snapshots and trades from Binance.
Persists raw JSON and compressed Parquet files.
"""

import ccxt
import time
import json
import pathlib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional


class CryptoDataCapture:
    """Capture order book snapshots and trades from crypto exchanges."""
    
    def __init__(self, exchange_name: str = 'binance', symbol: str = 'BTC/USDT', testnet: bool = True):
        """
        Initialize data capture.
        
        Args:
            exchange_name: Exchange to connect to (default: binance)
            symbol: Trading pair symbol (default: BTC/USDT)
            testnet: Use testnet endpoints (default: True for safe testing)
        """
        # Configure exchange with testnet support
        exchange_config = {'enableRateLimit': True}
        
        # Set testnet URLs for Binance Futures
        if testnet and exchange_name.lower() == 'binance':
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
            exchange_config['options'] = {'defaultType': 'future'}
        
        self.exchange = getattr(ccxt, exchange_name)(exchange_config)
        self.symbol = symbol
        self.testnet = testnet
        self.tick_dir = pathlib.Path('data/ticks')
        self.parquet_dir = pathlib.Path('data/parquet')
        self.tick_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
    def snapshot(self) -> Dict:
        """
        Capture single snapshot of order book and recent trades.
        
        Returns:
            Dict with timestamp, orderbook, and trades
        """
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=100)
            trades = self.exchange.fetch_trades(self.symbol, limit=200)
            ts = int(time.time() * 1000)  # millisecond timestamp
            
            return {
                'ts': ts,
                'datetime': datetime.fromtimestamp(ts/1000).isoformat(),
                'orderbook': ob,
                'trades': trades
            }
        except Exception as e:
            print(f"Error capturing snapshot: {e}")
            return None
    
    def save_snapshot_json(self, snapshot: Dict) -> pathlib.Path:
        """
        Save snapshot to JSON file.
        
        Args:
            snapshot: Snapshot data dictionary
            
        Returns:
            Path to saved file
        """
        if snapshot is None:
            return None
            
        ts = snapshot['ts']
        filepath = self.tick_dir / f"{ts}.json"
        filepath.write_text(json.dumps(snapshot, indent=2))
        return filepath
    
    def capture_continuous(self, duration_seconds: int = 3600, interval_ms: int = 1000):
        """
        Capture snapshots continuously for specified duration.
        
        Args:
            duration_seconds: How long to capture (default: 1 hour)
            interval_ms: Milliseconds between snapshots (default: 1000ms = 1s)
        """
        print(f"Starting continuous capture for {duration_seconds}s at {interval_ms}ms intervals")
        print(f"Symbol: {self.symbol}")
        
        start_time = time.time()
        count = 0
        
        try:
            while (time.time() - start_time) < duration_seconds:
                snapshot = self.snapshot()
                if snapshot:
                    self.save_snapshot_json(snapshot)
                    count += 1
                    if count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Captured {count} snapshots in {elapsed:.1f}s")
                
                # Sleep for interval
                time.sleep(interval_ms / 1000.0)
                
        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        
        elapsed = time.time() - start_time
        print(f"\nCapture complete: {count} snapshots in {elapsed:.1f}s")
        print(f"Saved to: {self.tick_dir}")
        
        return count
    
    def load_snapshots(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Load snapshots from JSON files.
        
        Args:
            limit: Maximum number of snapshots to load (None = all)
            
        Returns:
            List of snapshot dictionaries
        """
        json_files = sorted(self.tick_dir.glob("*.json"))
        if limit:
            json_files = json_files[:limit]
        
        snapshots = []
        for filepath in json_files:
            try:
                data = json.loads(filepath.read_text())
                snapshots.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return snapshots
    
    def snapshots_to_parquet(self, snapshots: List[Dict]) -> pathlib.Path:
        """
        Convert snapshots to Parquet format for efficient storage and querying.
        
        Args:
            snapshots: List of snapshot dictionaries
            
        Returns:
            Path to saved Parquet file
        """
        # Extract orderbook data
        ob_records = []
        for snap in snapshots:
            ts = snap['ts']
            ob = snap['orderbook']
            
            # Best bid/ask
            best_bid = ob['bids'][0] if ob['bids'] else [None, None]
            best_ask = ob['asks'][0] if ob['asks'] else [None, None]
            
            # Top 5 levels
            bid_prices_5 = [b[0] for b in ob['bids'][:5]]
            bid_sizes_5 = [b[1] for b in ob['bids'][:5]]
            ask_prices_5 = [a[0] for a in ob['asks'][:5]]
            ask_sizes_5 = [a[1] for a in ob['asks'][:5]]
            
            # Pad to 5 levels
            while len(bid_prices_5) < 5:
                bid_prices_5.append(None)
                bid_sizes_5.append(None)
            while len(ask_prices_5) < 5:
                ask_prices_5.append(None)
                ask_sizes_5.append(None)
            
            ob_records.append({
                'timestamp': ts,
                'datetime': snap['datetime'],
                'best_bid': best_bid[0],
                'best_bid_size': best_bid[1],
                'best_ask': best_ask[0],
                'best_ask_size': best_ask[1],
                'midprice': (best_bid[0] + best_ask[0]) / 2 if best_bid[0] and best_ask[0] else None,
                'spread': best_ask[0] - best_bid[0] if best_bid[0] and best_ask[0] else None,
                'bid_price_1': bid_prices_5[0], 'bid_size_1': bid_sizes_5[0],
                'bid_price_2': bid_prices_5[1], 'bid_size_2': bid_sizes_5[1],
                'bid_price_3': bid_prices_5[2], 'bid_size_3': bid_sizes_5[2],
                'bid_price_4': bid_prices_5[3], 'bid_size_4': bid_sizes_5[3],
                'bid_price_5': bid_prices_5[4], 'bid_size_5': bid_sizes_5[4],
                'ask_price_1': ask_prices_5[0], 'ask_size_1': ask_sizes_5[0],
                'ask_price_2': ask_prices_5[1], 'ask_size_2': ask_sizes_5[1],
                'ask_price_3': ask_prices_5[2], 'ask_size_3': ask_sizes_5[2],
                'ask_price_4': ask_prices_5[3], 'ask_size_4': ask_sizes_5[3],
                'ask_price_5': ask_prices_5[4], 'ask_size_5': ask_sizes_5[4],
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(ob_records)
        output_path = self.parquet_dir / f"orderbook_{snapshots[0]['ts']}_{snapshots[-1]['ts']}.parquet"
        df.to_parquet(output_path, compression='gzip', index=False)
        
        print(f"Saved {len(df)} orderbook snapshots to {output_path}")
        return output_path
    
    def load_parquet(self, filepath: pathlib.Path) -> pd.DataFrame:
        """Load orderbook data from Parquet file."""
        return pd.read_parquet(filepath)


def main():
    """Example usage: capture 1 hour of BTC/USDT data."""
    print("=" * 60)
    print("Crypto Market Data Capture")
    print("=" * 60)
    
    # Initialize capture
    capture = CryptoDataCapture(exchange_name='binance', symbol='BTC/USDT')
    
    # Test single snapshot
    print("\nTesting single snapshot...")
    snap = capture.snapshot()
    if snap:
        print(f"Timestamp: {snap['datetime']}")
        print(f"Best bid: {snap['orderbook']['bids'][0]}")
        print(f"Best ask: {snap['orderbook']['asks'][0]}")
        print(f"Recent trades: {len(snap['trades'])}")
        
        # Save test snapshot
        capture.save_snapshot_json(snap)
        print(f"Saved test snapshot")
    
    # Uncomment to capture continuous data
    # print("\nStarting continuous capture (1 hour)...")
    # capture.capture_continuous(duration_seconds=3600, interval_ms=1000)
    
    # Example: Convert saved snapshots to Parquet
    print("\nLoading saved snapshots...")
    snapshots = capture.load_snapshots(limit=10)
    if snapshots:
        print(f"Loaded {len(snapshots)} snapshots")
        parquet_path = capture.snapshots_to_parquet(snapshots)
        
        # Load and display
        df = capture.load_parquet(parquet_path)
        print("\nParquet data sample:")
        print(df[['datetime', 'midprice', 'spread', 'best_bid', 'best_ask']].head())


if __name__ == "__main__":
    main()
