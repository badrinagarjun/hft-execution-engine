# Algorithmic Trading Execution System

A comprehensive quantitative trading system for optimal order execution using microstructure analysis, machine learning, and adaptive execution strategies.

## Overview

This project implements a complete pipeline for algorithmic execution research:

1. **Data Capture**: Real-time cryptocurrency market data (order books + trades)
2. **LOB Simulator**: Limit order book matching engine for controlled testing
3. **Execution Schedules**: TWAP, VWAP, and POV algorithms
4. **Order-Flow Features**: Microstructure signals (imbalance, depth, etc.)
5. **ML Prediction**: Short-term price direction forecasting
6. **Adaptive Execution**: Signal-driven execution policies
7. **Performance Metrics**: Implementation Shortfall, slippage, statistical tests
8. **Backtesting Framework**: Event-driven simulator

## Project Structure

```
quant/
├── data/                      # Market data storage
│   ├── ticks/                # Raw JSON tick data
│   └── parquet/              # Compressed Parquet files
├── data_capture.py           # Crypto data capture (ccxt)
├── lob_simulator.py          # Limit order book simulator
├── execution_schedules.py    # TWAP, VWAP, POV algorithms
├── orderflow_features.py     # Microstructure feature extraction
├── ml_predictor.py           # ML models for direction prediction
├── performance_metrics.py    # IS, slippage, statistical tests
├── backtester.py            # Event-driven backtesting engine
├── experiments.ipynb         # Main experiments notebook
└── README.md                # This file
```

## Installation

### 1. Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies:

```powershell
pip install ccxt pandas numpy scipy scikit-learn matplotlib numba jupyterlab
```

## Quick Start

### 1. Capture Market Data

Capture live order book snapshots and trades from Binance:

```python
from data_capture import CryptoDataCapture

# Initialize capture
capture = CryptoDataCapture(exchange_name='binance', symbol='BTC/USDT')

# Capture 1 hour of data (1 snapshot per second)
capture.capture_continuous(duration_seconds=3600, interval_ms=1000)

# Convert to Parquet for efficient storage
snapshots = capture.load_snapshots()
capture.snapshots_to_parquet(snapshots)
```

### 2. Test LOB Simulator

```python
from lob_simulator import SimpleLOB

# Create order book
lob = SimpleLOB()

# Add limit orders
lob.add_limit('buy', 50000.0, 10.0)
lob.add_limit('sell', 50001.0, 8.0)

# Execute market order
fills = lob.market_buy(5.0)
print(f"Executed: {fills}")
```

### 3. Run Execution Strategies

```python
from execution_schedules import twap_schedule, vwap_schedule

# TWAP: equal slices
twap = twap_schedule(total_qty=10000, periods=20)

# VWAP: volume-weighted
volume_profile = [100, 120, 110, 130, ...]  # Historical volumes
vwap = vwap_schedule(total_qty=10000, volume_profile=volume_profile)
```

### 4. Extract Order-Flow Features

```python
from orderflow_features import extract_features_from_snapshot

# Extract features from snapshot
features = extract_features_from_snapshot(snapshot)
print(features['imbalance_5'])  # Top 5 level imbalance
print(features['spread'])        # Bid-ask spread
print(features['trade_imbalance'])  # Recent trade flow
```

### 5. Train ML Predictor

```python
from ml_predictor import OrderFlowPredictor
from orderflow_features import create_feature_matrix

# Create features from snapshots
feature_df = create_feature_matrix(snapshots)

# Train predictor
predictor = OrderFlowPredictor(model_type='logistic', horizon=1)
results = predictor.train(feature_df)

# Predict on new data
proba = predictor.predict_proba(features)  # P(price increase)
```

### 6. Run Backtest

```python
from backtester import Backtester, TWAPStrategy, AdaptiveStrategy
from execution_schedules import twap_schedule

# Create backtester
backtester = Backtester(snapshots, bucket_duration_ms=10000)

# Test TWAP
schedule = twap_schedule(total_qty=100, periods=10)
strategy = TWAPStrategy(total_qty=100, schedule=schedule, side='buy')
results = backtester.run(strategy)

print(f"Implementation Shortfall: {results['is_bps']:.2f} bps")
print(f"Fill Rate: {results['fill_rate']:.2f}%")
```

## Experiments

### Experiment 1: TWAP vs VWAP Baseline

Compare equal-weighted (TWAP) vs volume-weighted (VWAP) execution:

```python
# Run N times with different market regimes
n_runs = 50
twap_is = []
vwap_is = []

for run in range(n_runs):
    # TWAP
    twap_result = run_backtest(TWAPStrategy(...))
    twap_is.append(twap_result['is_bps'])
    
    # VWAP
    vwap_result = run_backtest(VWAPStrategy(...))
    vwap_is.append(vwap_result['is_bps'])

# Statistical test
from performance_metrics import paired_t_test
test = paired_t_test(twap_is, vwap_is)
print(f"p-value: {test['p_value']:.6f}")
```

### Experiment 2: Signal-Enhanced Execution

Add ML prediction to execution decisions:

```python
# Train predictor
predictor = OrderFlowPredictor(model_type='logistic')
predictor.train(training_features)

# Create adaptive strategy
def predict_fn(features):
    return predictor.predict_proba(features)

strategy = AdaptiveStrategy(
    total_qty=100,
    schedule=vwap_schedule,
    side='buy',
    predictor=predict_fn,
    aggressive_threshold=0.65  # Be aggressive if P(up) > 0.65
)

result = backtester.run(strategy)
```

### Experiment 3: POV Sensitivity

Test different participation rates:

```python
from execution_schedules import pov_schedule

rates = [0.005, 0.01, 0.02, 0.03, 0.05]
results = []

for rate in rates:
    schedule = pov_schedule(total_qty=100, observed_volumes=volumes, 
                          participation_rate=rate)
    strategy = POVStrategy(...)
    result = backtester.run(strategy)
    results.append({'rate': rate, 'is_bps': result['is_bps']})
```

## Key Metrics

### Implementation Shortfall (IS)
```
IS = (avg_exec_price - arrival_price) * sign(direction)
```
Measures execution cost relative to price at decision time.

### VWAP Slippage
```
Slippage = avg_exec_price - benchmark_VWAP
```
Measures performance vs volume-weighted average price.

### Market Impact
- **Temporary Impact**: Execution price vs arrival price
- **Permanent Impact**: Post-execution price change
- **Total Impact**: Sum of temporary + permanent

### Fill Rate
```
Fill Rate = executed_quantity / scheduled_quantity * 100%
```

## Performance Tips

1. **Data Storage**: Use Parquet for fast loading (10-100x faster than JSON)
2. **Feature Engineering**: Compute features once, save to disk
3. **Backtesting**: Use vectorized operations where possible
4. **LOB Updates**: Only recompute changed levels
5. **Memory**: Process data in chunks for large datasets

## Advanced Features

### Numba Acceleration (Optional)

For high-frequency simulation, use Numba JIT compilation:

```python
from numba import jit

@jit(nopython=True)
def fast_imbalance_calc(bids, asks):
    # Accelerated computation
    ...
```

### Walk-Forward Analysis

Train on expanding window:

```python
train_periods = [(0, 1000), (0, 2000), (0, 3000), ...]
for train_start, train_end in train_periods:
    # Train on [train_start:train_end]
    # Test on [train_end:train_end+test_size]
    ...
```

### Market Regime Detection

Classify market states (high/low volatility, trending/mean-reverting):

```python
from sklearn.cluster import KMeans

# Extract regime features
regime_features = df[['volatility', 'trend', 'volume']].values
kmeans = KMeans(n_clusters=3)
regimes = kmeans.fit_predict(regime_features)

# Run strategies per regime
for regime in [0, 1, 2]:
    subset = df[regimes == regime]
    # Backtest on regime subset
```

## Testing

Each module has built-in tests. Run individually:

```bash
python data_capture.py       # Test data capture
python lob_simulator.py       # Test LOB simulator
python execution_schedules.py # Test schedules
python orderflow_features.py  # Test features
python ml_predictor.py        # Test predictor
python performance_metrics.py # Test metrics
python backtester.py          # Test backtester
```

## Research Extensions

1. **Hawkes Processes**: Replace Poisson arrivals with self-exciting processes
2. **Queue Position**: Model exchange queue priority
3. **Fees & Rebates**: Include maker/taker fee structures
4. **Multi-Asset**: Extend to portfolio execution
5. **RL Agents**: Train reinforcement learning policies
6. **HFT Strategies**: Sub-second execution tactics
7. **Equity Markets**: Use LOBSTER or Polygon data
8. **Options**: Add derivatives execution

## References

- Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions."
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order book events."
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and High-Frequency Trading."
- Gatheral, J. (2010). "No-dynamic-arbitrage and market impact."

## License

MIT License - See LICENSE file for details

## Contact

For questions or contributions, please open an issue on GitHub.

---

**Note**: This is a research/educational framework. Use appropriate risk controls for live trading.
