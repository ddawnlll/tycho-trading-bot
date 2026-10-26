"""Single source of truth for the strategy that research validated and the live runner trades.

Provenance (see runs/):
  * signal params: IS-plateau center of portfolio_grid (runs/*_portfolio_grid_donchian_rg1200)
  * portfolio constraints: IS pick by MAR in multi_research (runs/*_multi_research) was
    max_positions=5, max_gross=3.0, risk=0.01. Live deliberately tightens gross to 2.0 (the cap
    rarely binds: identical IS/OOS metrics at risk 0.005) and defaults risk to 0.005.
"""
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"]
TIMEFRAME = "4h"
STRATEGY = "donchian"
PARAMS = dict(entry_n=30, exit_n=30, stop_atr=3.0, allow_short=False)
REGIME_N = 1200          # BTC close vs EMA(1200 x 4h ~ 200 days): longs only in BTC bull regime
MAX_POSITIONS = 5
MAX_GROSS = 2.0          # total notional / equity
MAX_LEV_POS = 1.0        # per-position notional / equity
RISK_IS_PICK = 0.01      # what research selected
RISK_DEFAULT = 0.005     # live default: half risk until live fills are confirmed to match backtest
WARMUP_BARS = 300        # per-symbol history for Donchian/ATR (ATR EWM init influence ~e^-15)
REGIME_WARMUP_BARS = 4500  # BTC history for EMA(1200): init influence (1-2/1201)^4500 ~ 0.06%
