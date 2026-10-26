# Momentum candidates v1: results (2026-10-27)

**Verdict: no candidate survived the pre-registered validation. Nothing was activated.**
Paper trading continues with the two existing experts; real-money execution remains structurally
disabled (runner modes are `paper`/`sandbox` only, no exchange execution client is constructed).

Protocol: `research/momentum_protocol.json`, committed in `a5ebfea` before any backtest. Amendments
(`d37dfd4`, `f31c2ab`) were recorded before the holdout run; the second was committed after the
corrected dev Nautilus run had been seen. Every trial, including failed and aborted
ones, is in the hash-chained `research/momentum_trials.jsonl`. The frozen prospective window
(2026-10-26 20:00 UTC → 2027-04-24 20:00 UTC) was never read: dev data ends 2026-10-26 12:00 and the
holdout archives end 2026-09-30.

## Candidates (hypotheses drawn from championship traders, not assumed alpha)
| expert | source idea | rule |
|---|---|---|
| leader_breakout | Luk / Tito / Minervini | top momentum-rank coin, BTC bull + breadth > 30%, breakout from a tight base, 1.5 ATR stop, 3 ATR trail |
| tsmom | trend CTAs / L. Williams mechanical trend | daily vote over 30/90/180-day returns; enter when all 3 > 0, exit when the sum ≤ 0; 3–5 ATR stop |
| dual_momentum | J Law regime switch / Antonacci | weekly; rank ≤ k AND 90-day return > 0; exit on rank or absolute momentum |
| vol_expansion | L. Williams volatility breakout | expansion bar (body > k·ATR, close in top 20%) above EMA50 in a bull regime |
| trend_long | current bot (Donchian 30/30/3) | incumbent baseline, fixed parameters |

Sizing for all: 0.5% risk/trade, ≤5 positions, ≤2× gross, ≤1× per coin, one expert per coin.
Costs: 5 bp taker fee + 2/5 bp slippage (dev), 10 bp slippage (holdout alts), real funding, 3× stress.
Tradability is point-in-time: 30-day median daily quote volume ≥ $20M, ≥200 bars of history; momentum
rank and breadth are computed only among coins tradable at that moment.

## 1. Walk-forward on the dev universe — HISTORICAL DIAGNOSTIC
12 coins, annual refit on prior years only, OOS 2021-01 → 2026-10 (tycho engine, `runs/20261026-212449_momentum_wf`).
Dev data was used to design earlier experts, so this is not clean out-of-sample evidence.

| expert | CAGR | Sharpe | max DD | trades | 3× cost CAGR | WF criteria |
|---|---|---|---|---|---|---|
| leader_breakout | 8.9% | 0.58 | 38.4% | 794 | 0.9% | fail (DD) |
| tsmom | 23.6% | 0.72 | 56.1% | 137 | 22.3% | fail (DD) |
| dual_momentum | 6.2% | 0.74 | 11.5% | 165 | 5.6% | pass |
| vol_expansion | 6.4% | 0.55 | 20.6% | 670 | 1.1% | pass |
| trend_long (incumbent) | 21.9% | 1.14 | 19.9% | 537 | 18.0% | pass* |
| BTC buy & hold | 18.7% | 0.59 | 76.7% | – | – | baseline |
| 12-coin basket buy & hold | 47.9% | 0.90 | 78.8% | – | – | baseline (survivorship) |

*trend_long's parameters were chosen on this data earlier; its walk-forward is not independent.

NautilusTrader re-run of the final parameters on the same period, with funding simulated **inside the
venue** (`runs/20261026-213424_momentum_nautilus_dev`, all tycho/Nautilus gaps ≤ 1%): trend_long Sharpe
1.16, tsmom 0.77, vol_expansion 0.78, dual_momentum 0.73, leader_breakout 0.72. None has a block-bootstrap
lower bound clearly above 0 together with null p ≤ 0.01 (best: trend_long, p = 0.021).

## 2. Untouched holdout A — 538 other Binance perps incl. delisted, 2021-01 → 2026-09
Never used before this protocol (`runs/20261026-214401_momentum_holdout`). **Genuine out-of-sample
evidence**, with survivorship bias reduced but not removed: 20 tradable series end before 2026-09-30
(delisted, halted or renamed): AERGO, AKRO, ANT, AUDIO, BTS, BZRX, COCOS, DODO, EOS, FOOTBALL, FRONT,
GAL, HNT, LUNA, MATIC, RNDR, SRM, SXP, TOMO, YFII (MATIC, RNDR, EOS, GAL are renames/migrations, not
failures). FTT is included. Perps with fewer than 12 monthly archive files were not downloaded (see
"Remaining uncertainties").

| expert | CAGR | Sharpe | max DD | trades | 3× cost total | bootstrap lower | null p | verdict |
|---|---|---|---|---|---|---|---|---|
| tsmom | 9.4% | 0.44 | 44.2% | 330 | +54% | < 0 | 0.13 | REJECT |
| trend_long (incumbent) | 6.7% | 0.38 | 49.8% | 968 | +10% | < 0 | 0.22 | REJECT |
| leader_breakout | 1.7% | 0.21 | 28.3% | 458 | −9% | < 0 | 0.31 | REJECT |
| dual_momentum | −0.5% | −0.07 | 18.0% | 222 | −5% | < 0 | 0.58 | REJECT |
| vol_expansion | −16.5% | −0.84 | 72.7% | 1990 | −83% | < 0 | 0.99 | REJECT |
| *context, not pre-registered:* BTC buy & hold | 20.3% | 0.61 | 76.7% | – | – | – | – | – |
| *context:* equal-weight liquid alt basket | −13.3% | 0.30 | 96.3% | – | – | – | – | – |

## 3. Untouched holdout B — Binance spot 2018-01 → 2019-12 (pre-perpetual era, 2018 bear)
All five stayed within limits (the BTC regime gate kept them mostly in cash in 2018): leader_breakout
+11.9% (DD 6.9%), tsmom +12.5% (DD 19.9%), trend_long +12.0% (DD 9.5%), vol_expansion +8.0%,
dual_momentum +3.5%. Few trades (18–75), so weak evidence either way.

## What this means
- **No candidate shows convincing, repeatable edge.** On untouched data the best trend strategies
  (tsmom, trend_long) made positive net returns and were far better than holding alts (−13%/yr), but did
  not beat simply holding BTC on a risk-adjusted basis and were not statistically distinguishable from
  noise (null p 0.13–0.22, bootstrap lower bound < 0), with 44–50% drawdowns at 0.5% risk.
- Short-horizon breakout ideas (vol_expansion, leader_breakout) degrade badly on less liquid alts.
  Hypothesis, not measured: many small losses plus 10 bp slippage overwhelm the occasional winner.
- The incumbent trend_long looked strong on the dev data (Sharpe 1.14–1.16) and weak on holdout A
  (0.38). Part of that gap is plausibly selection/design bias, but the universes and costs differ
  (majors at 2 bp vs alts at 10 bp slippage), so its size is not established.
- Per protocol there is **no re-tuning on holdout data**. A new hypothesis needs a new pre-registration
  and fresh data; the only remaining untouched data is the frozen prospective window.

## Failed / aborted experiments (all in the ledger)
- First dev Nautilus run (`20261026-212815`): INVALID. A funding overlay applied after the fact was
  +11% optimistic because Nautilus kept sizing on pre-funding equity. Fixed with an in-venue
  `FundingModule` (`tycho/nautilus_funding.py`, tested).
- Nautilus confirmation on holdout A was attempted twice for tsmom and stopped by a free-memory
  watchdog (1.13 GB, then 1.02 GB free) to protect the running paper bot. Holdout verdicts therefore
  rest on the tycho engine, whose parity with Nautilus (≤ 1% final equity, in-venue funding) was
  measured on the dev period, not on the holdout itself.

## Remaining uncertainties
- The downloader skipped perps with fewer than 12 monthly archive files. That rule uses hindsight
  (a coin's lifetime is only known afterwards) and drops exactly the perps that listed and died within
  a year, so survivorship bias is reduced, not eliminated.
- Holdout A momentum rank includes BTC (regime source, not traded), which can occupy a top-k slot; the
  rank also lacks the protocol's 200-bar history rule (liquidity gate only). Affects leader_breakout and
  dual_momentum; not re-run (see `docs/LEFT_TASKS.md`).
- A position in a coin whose data ends (delisting) stays open, marked at the last close, until the end
  of the run in the frozen engine; whether this occurred in holdout A was not measured.
- tycho/Nautilus parity (≤ 1%) was measured on the dev coins only; dual_momentum's gap was 0.998%.
  FundingModule charges positions open at bar close, tycho those open at bar open, so a position
  stopped inside a funding bar escapes that bar's funding in Nautilus.
- The context-baseline rows in the holdout table and the two aborted Nautilus attempts were produced
  by ad-hoc code that is not in the repository; their ledger rows are not reproducible as-is.
- `runs/index.csv` records HEAD at run time; some runs (212449, 212815, 213424) ran with uncommitted
  changes on top of that commit.
- Funding uses the bar open as a mark proxy, not the settlement mark tape. Bar fills don't prove
  exchange fill quality.
- 30-day bootstrap blocks may be short for trend trades that last months.
- Holdout A shares its BTC regime series with the dev data (BTC itself is not traded there).
