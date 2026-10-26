# Run log

Every research/backtest invocation writes `runs/<UTC timestamp>_<tag>/` (config, summary, tables)
and appends one line to `index.csv`. Runs are never deleted, including bad ones.

## Errata
- `20261026-145242_trend_pullback_BTCUSDT_4h`, `20261026-145251_donchian_BTCUSDT_4h`: the
  buy-and-hold baseline in these summaries is wrong (0.0%): the B&H entry signal only existed
  on the first bar of the full series, so the sliced IS/OOS run never entered. Fixed in
  `buy_hold` (enter whenever flat) and the baseline now uses spot (no funding). Strategy
  numbers in those runs are unaffected.
- `20261026-153250_expert_research` vs `20261026-153423_expert_research`: both are stamped with
  git `4b0134c`, but the second ran after **uncommitted** MarketState fixes from an architecture
  review (breadth only counts coins with >= 1000 bars; percentiles need a full 1-year window).
  The fixes were made for correctness, not from OOS numbers, but OOS columns were visible in the
  first run, and `squeeze_breakout` flipped from admitted to rejected between the two runs.
  Treat the expert selection as "IS-only by rule, not blind". The 2024+ OOS window has been
  viewed many times across this research; it is no longer a clean holdout.
- `20261026-153838_nautilus_crosscheck`: ran with the original Nautilus stop logic
  (`modify_order`), which Binance rejects for STOP_MARKET. Superseded by
  `20261026-160403_nautilus_crosscheck` (cancel/replace stops; per-trade exit + price checks).
- `20261026-212815_momentum_nautilus_dev`: INVALID (tycho/Nautilus gaps 2–11%). Funding was an
  after-the-fact overlay, so Nautilus kept sizing on pre-funding equity. Superseded by
  `20261026-213424_momentum_nautilus_dev` (funding simulated inside the venue; all gaps <= 1%).
