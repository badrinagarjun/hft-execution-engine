# Binance Testnet Guide

## What is the Testnet?

The Binance Futures Testnet is a **risk-free testing environment** that mirrors the real Binance Futures platform. You can:
- ✅ Test trading strategies without risking real money
- ✅ Get real-time market data
- ✅ Practice order execution
- ❌ **NOT** trade with real funds (testnet uses fake USDT)

---

## Testnet Configuration

The system is now configured to use Binance Futures Testnet by default:

- **REST API Base URL**: `https://testnet.binancefuture.com/fapi/v1`
- **WebSocket Base URL**: `wss://fstream.binancefuture.com`

### In Code:

```python
# Use testnet (default, safe for testing)
capture = CryptoDataCapture(exchange_name='binance', symbol='BTC/USDT', testnet=True)

# Use production (requires real API keys and trades real money)
capture = CryptoDataCapture(exchange_name='binance', symbol='BTC/USDT', testnet=False)
```

---

## Getting Testnet API Keys (Optional)

For **read-only** operations (fetching order books, trades), you **don't need API keys**.

For **placing orders** on testnet:

1. **Register** at: https://testnet.binancefuture.com/
2. **Login** with your account
3. **Generate API Key**:
   - Go to API Management
   - Create new API key
   - Save your API Key and Secret Key
4. **Use in code**:
   ```python
   exchange = ccxt.binance({
       'apiKey': 'YOUR_TESTNET_API_KEY',
       'secret': 'YOUR_TESTNET_SECRET',
       'enableRateLimit': True,
       'urls': {
           'api': {
               'public': 'https://testnet.binancefuture.com/fapi/v1',
               'private': 'https://testnet.binancefuture.com/fapi/v1',
           }
       },
       'options': {'defaultType': 'future'}
   })
   ```

---

## Rate Limits

The testnet has the same rate limits as production:

- **Weight-based limits**: Each endpoint consumes "weight"
- **Order limits**: Max orders per time window
- **IP-based**: Limits apply per IP address

**Best Practice**: Use WebSocket streams for real-time data instead of polling REST API.

---

## Switching to Production

⚠️ **WARNING**: Only switch to production when you're ready to trade with **real money**.

### Steps:

1. **Get real Binance API keys** from https://www.binance.com/
2. **Update code**:
   ```python
   capture = CryptoDataCapture(
       exchange_name='binance', 
       symbol='BTC/USDT', 
       testnet=False  # ⚠️ REAL MONEY
   )
   ```
3. **Add API credentials** if placing orders:
   ```python
   exchange = ccxt.binance({
       'apiKey': 'YOUR_REAL_API_KEY',
       'secret': 'YOUR_REAL_SECRET',
       'enableRateLimit': True,
       'options': {'defaultType': 'future'}
   })
   ```

---

## Security Best Practices

### For Testnet:
- ✅ Safe to experiment freely
- ✅ No real money at risk
- ✅ Share code without concerns

### For Production:
- ❌ **NEVER** commit API keys to Git
- ❌ **NEVER** share your secret key
- ✅ Use environment variables:
  ```python
  import os
  api_key = os.environ.get('BINANCE_API_KEY')
  secret = os.environ.get('BINANCE_SECRET')
  ```
- ✅ Restrict API permissions (read-only if possible)
- ✅ Set IP whitelist in Binance settings

---

## Current System Status

✅ **Testnet Mode Enabled** (Safe for Research)

The system will:
1. Connect to Binance Testnet for live data capture
2. Use synthetic data generation for backtesting
3. Never risk real money

---

## FAQ

**Q: Do I need API keys for backtesting?**  
A: No! The backtester uses historical/synthetic data. No exchange connection needed.

**Q: Can I capture real market data without API keys?**  
A: Yes! Public endpoints (order books, trades) don't require authentication.

**Q: How do I know if I'm on testnet vs production?**  
A: Check the URL in error messages or add logging:
```python
print(f"Exchange URL: {capture.exchange.urls}")
print(f"Testnet mode: {capture.testnet}")
```

**Q: Is testnet data realistic?**  
A: Yes! It mirrors real market data, but with slightly lower liquidity.

---

## Next Steps

1. ✅ **Test data capture** - Run cell 3 in `experiments.ipynb`
2. ✅ **Run backtests** - Use synthetic data (no API needed)
3. ✅ **Analyze results** - Compare TWAP vs VWAP vs Adaptive
4. 📚 **Study the code** - Understand execution algorithms
5. 🚀 **Extend the system** - Add your own strategies

---

**Remember**: This is a **research framework**, not a production trading bot. Always backtest thoroughly before considering live trading!
