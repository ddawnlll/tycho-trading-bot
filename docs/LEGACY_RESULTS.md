> Archived pre-migration notes. Commands, admissions and performance claims below are historical, superseded by the root README. They do not authorize exchange execution or constitute current evidence.

# Tycho — crypto expert-ensemble bot (Binance USDT-M perpetuals)

Research → experts that trade only in the market state they request → the same rules in a
tycho runner and a NautilusTrader node. Every research invocation is saved under `runs/`
(see `runs/README.md`, `runs/index.csv`).

## Architecture
- `tycho/marketstate.py` — **MarketState service**, causal, computed at each 4h close:
  BTC regime (close vs EMA1200 ≈ 200 days), breadth (share of coins above EMA200), BTC vol
  percentile, per-coin trend/range (EMA50/200 + efficiency ratio), funding z-score, 30-day
  momentum rank.
- `tycho/experts.py` — **experts**: each = signal rule + `requires(state)`; entries only where the
  requested state holds. One expert per coin at a time (Binance one-way netting); priority = order
  in `config.EXPERTS`.
- `tycho/engine_multi.py` — shared-account backtester (caps: 5 positions, 2x gross, 1x per coin).
- `tycho/nautilus_bt.py` — the same ensemble as a NautilusTrader `Strategy` (backtest).
- `tycho/nautilus_live.py` — NautilusTrader `TradingNode` (sandbox / Binance demo / live).
- `tycho/live.py` — lightweight ccxt runner (paper / demo / live) with the same decisions.
- `tycho/config.py` — single source of truth for universe, experts, sizing.

## Experts (selection on 2020–2023 only; OOS = 2024-01 → 2026-10, never used to select)
| expert | requested state | IS Sharpe | admitted | OOS Sharpe |
|---|---|---|---|---|
| trend_long (Donchian 30/30, 3 ATR) | BTC bull | 1.42 | yes | 0.93 |
| xs_momentum (top-4 by 30d return, daily) | BTC bull + breadth > 50% | 1.77 | yes | 0.33 |
| funding_squeeze (shorts crowded) | BTC bull + funding z < −1 | 0.78 | yes | −0.44 |
| squeeze_breakout (vol contraction) | low vol + bull | 0.36 | no | 0.22 |
| trend_short (Donchian short) | BTC bear + weak breadth | 0.28 | no | −0.57 |
| range_revert (RSI2 / Bollinger) | coin in range + bull | −0.59 | no | −2.06 |

Ensemble pick on IS (by MAR): **trend_long + xs_momentum, max 5 positions** (funding_squeeze lost
OOS in every combination and was not picked).

## Results (0.5% risk/trade, fees 0.05%/side, slippage 2–5 bps, real funding)
| | CAGR | max DD | Sharpe |
|---|---|---|---|
| Ensemble, OOS 2024–26 | 16.6% | −19.3% | 0.95 |
| Ensemble, OOS, 3× costs | 12.9% | −20.0% | 0.77 |
| trend_long alone, OOS | 16.2% | −18.8% | 0.93 |
| BTC spot buy & hold, OOS | 15.9% | −60.7% | 0.51 |

Ensemble per year: 2020 +52%, 2021 +85%, 2022 −2%, 2023 +20%, 2024 +46%, **2025 +2%, 2026 YTD −1%**.
46% of months positive. 3-month block bootstrap: P(losing 12 months) ≈ 18%, 95th-pct DD ≈ −24%.
**Adding experts barely improved on trend_long alone** — the honest headline.

## NautilusTrader cross-check (`runs/*_nautilus_crosscheck`)
Same rounded bars, slippage 0, funding 0, both engines: BTC 44/44 trades, 12-coin trend 286/286,
ensemble 252/252 — matched on entry AND exit time; per-trade entry price diff p99 0.03%, exit p99
0.01%; final equity within 0.03%, max daily gap ≤ 0.15%. Trade *timing* agreement is largely implied
by the shared signals; the independent part is Nautilus' fills, stop triggers, fees and accounting.
Nautilus order handling (stop cancel/replace, full-fill stop placement, denied entry) is also tested
on synthetic data in `tests/test_nautilus_orders.py`.

## Run
```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/Scripts/python.exe numpy pandas pyarrow requests xgboost scikit-learn ccxt pytest pyflakes nautilus_trader
.venv/Scripts/python.exe scripts/download.py                  # ~5 min, public data
curl -s https://fapi.binance.com/fapi/v1/exchangeInfo -o data/exchange_info.json
.venv/Scripts/python.exe -m pytest -q tests                   # 21 tests, ~4 min
.venv/Scripts/python.exe -m scripts.expert_research           # per-expert IS/OOS
.venv/Scripts/python.exe -m scripts.ensemble_research         # ensemble pick on IS
.venv/Scripts/python.exe -m scripts.nautilus_crosscheck       # tycho vs NautilusTrader
.venv/Scripts/python.exe -m scripts.validate --risk 0.005     # stress tests
.venv/Scripts/python.exe -m scripts.nautilus_live             # Nautilus sandbox: live data, simulated fills
.venv/Scripts/python.exe -m scripts.live                      # tycho paper runner
```
Demo / live need `BINANCE_API_KEY` / `BINANCE_API_SECRET`:
`-m scripts.nautilus_live --mode testnet` (Binance demo account) or `--mode live --i-understand-real-money`
(same flags for `-m scripts.live`). Journals: `journal/`; Nautilus logs MARKET_STATE each 4h bar.

## Known limitations
- Backtest ≠ future. OOS is one 34-month window; 2025–26 were roughly flat.
- The 12-coin universe was picked in 2026 → survivorship bias, also in breadth/momentum rank.
- Many variants were tried (all logged); IS-best parameters repeatedly did worse OOS than average.
- Real-exchange order paths (tycho ccxt and Nautilus Binance exec client) were **never executed** —
  no keys were used. Nautilus sandbox and backtest exercised the Nautilus order logic (market entry,
  reduce-only STOP_MARKET, modify, cancel) on Nautilus' simulated exchange only. Run demo first.
- Nautilus live: bar fetch for decisions is a blocking REST call (~15–30 s every 4 h) inside the
  event loop. Restart safety: last decided bar is persisted, bars that closed > 30 min ago are
  skipped, reconciled orders are claimed, and any position without a saved owner and exactly one
  open stop has its orders cancelled and is flattened. Stops move by submit-new-then-cancel-old
  (Binance rejects modify on STOP_MARKET) — verified only on Nautilus' simulated exchange.
- Selection was IS-only by rule, but OOS results were visible throughout and code changed between
  two expert runs (see `runs/README.md` errata). The 2024+ window is no longer a clean holdout.
- tycho live: exit price after an exchange stop is logged as an estimate; stops trigger on last price.
- Funding is in tycho backtests but not in Nautilus (not simulated there).
- BNB funding history has many exact-zero records → BNB long costs may be slightly understated.
