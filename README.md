# Tycho — crypto trend bot (Binance USDT-M perpetuals)

Research → one validated strategy → the same rules in a paper/testnet/live runner.
Every research invocation is saved under `runs/` (see `runs/README.md`, `runs/index.csv`).

## What trades
`tycho/config.py` is the single source of truth (backtest and live both read it):

- Universe: BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC DOT TRX (USDT-M perps), 4h bars
- Entry: long when the 4h close breaks the prior 30-bar high, **only while BTC > EMA(1200 x 4h ≈ 200 days)**
- Stop: entry-bar close − 3×ATR(20), trailed up to the 30-bar low; exchange-side stop order
- Sizing: risk per trade = 0.5% of equity (research pick was 1%), ≤1x notional per position,
  ≤2x gross, ≤5 concurrent positions
- No shorts, no ML filter (both tested, both worse out-of-sample — see below)

## Results (fees 0.05%/side, slippage 2–5 bps, real funding; IS 2020–2023, OOS 2024-01 → 2026-10)

| config | period | CAGR | max DD | Sharpe |
|---|---|---|---|---|
| risk 0.5% | OOS 2024–26 | 16.2% | −18.8% | 0.93 |
| risk 0.5% | OOS, 3× costs | 11.8% | −20.1% | 0.72 |
| risk 1.0% | OOS 2024–26 | 28.7% | −31.2% | 0.92 |
| 12-coin equal-weight buy & hold | OOS 2024–26 | 7.0% | −61.0% | 0.38 |
| BTC spot buy & hold | OOS 2024–26 | 15.9% | −60.7% | 0.51 |

Per year at 0.5% risk: 2020 +66%, 2021 +75%, 2022 −0.8%, 2023 +20%, 2024 +48%, **2025 +5.3%, 2026 YTD −1.4%**.
Only ~42% of months are positive; returns come from a few trending months. Block-bootstrap
(3-month blocks): P(losing 12 months) ≈ 15%, 95th-pct drawdown ≈ −21% at 0.5% risk (−38% at 1%).

## What was tried and rejected (all runs in `runs/`)
- **Martin-Luk-style first-pullback entries**: worked weakly on BTC alone, **lost money OOS** on the
  12-coin portfolio in every variant (−3% to −12% CAGR).
- **MA-trend (50/200) time-series momentum**: ~0–5% OOS CAGR.
- **Long/short Donchian**: worse than long-only.
- **XGBoost meta-label filter** (walk-forward, no leakage): OOS CAGR 16.4% → 11.1%; it filtered out
  the rare big winners (+102R dropped in 2024) that trend following lives on.
- Param optimisation itself: IS-best params ranked *worse* than median OOS (Spearman −0.28), so the
  live params are the centre of the IS plateau, not the IS peak. 120/120 Donchian param sets were
  OOS-profitable → the edge is the family (trend + regime), not a magic number.

## Run
```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/Scripts/python.exe numpy pandas pyarrow requests xgboost scikit-learn ccxt pytest
.venv/Scripts/python.exe scripts/download.py                 # ~5 min, public data
.venv/Scripts/python.exe -m pytest -q tests                  # engine, causality, live==backtest replay
.venv/Scripts/python.exe -m scripts.validate --risk 0.005    # re-run stress tests (saved to runs/)
.venv/Scripts/python.exe -m scripts.live                     # paper trading loop (no keys)
```
Testnet/live need `BINANCE_API_KEY` / `BINANCE_API_SECRET` in the environment:
`python -m scripts.live --mode testnet` (Binance demo account) and
`python -m scripts.live --mode live --i-understand-real-money`. Journal: `journal/<mode>_YYYYMM.jsonl`.

## Known limitations
- Backtest ≠ future. OOS is one 34-month window; 2025–26 were roughly flat.
- testnet/live order paths (ccxt market orders, `stopLossPrice` reduce-only stops routed to
  Binance's Algo Order API, demo trading switch) were checked against the ccxt 4.5.90 source but
  **never executed** — no keys were used. Unverified: `reduceOnly` acceptance on algo orders,
  `totalMarginBalance` on the demo account, one-way/cross margin mode (assumed, not set). Run
  testnet first.
- Safety behaviour (tested with a fake broker in `tests/test_live_cycle.py`): new stop is placed
  before the old one is cancelled; if an entry's stop can't be placed the position is flattened;
  state is saved after every fill; a bar is never processed twice (restart-safe); stale bars are
  not traded; testnet/live refuse to start if state and exchange positions disagree.
- Stops trigger on last price in live (CONTRACT_PRICE) like the backtest's bar lows; real
  fills on fast moves will be worse than the stop price.
- Live exit price after an exchange stop is logged as an estimate (last price), not the fill.
- The bot starts flat; it does not adopt positions the backtest would already hold.
- BNB funding history has many exact-zero records (can't distinguish from missing) → BNB long
  costs may be slightly understated.
- Sub-account research (`portfolio.py`) and shared-account engine (`engine_multi.py`) differ; live
  matches the shared-account engine (verified by `tests/test_live.py`).
