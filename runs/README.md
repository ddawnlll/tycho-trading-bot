# Run log

Every research/backtest invocation writes `runs/<UTC timestamp>_<tag>/` (config, summary, tables)
and appends one line to `index.csv`. Runs are never deleted, including bad ones.

## Errata
- `20261026-145242_trend_pullback_BTCUSDT_4h`, `20261026-145251_donchian_BTCUSDT_4h`: the
  buy-and-hold baseline in these summaries is wrong (0.0%): the B&H entry signal only existed
  on the first bar of the full series, so the sliced IS/OOS run never entered. Fixed in
  `buy_hold` (enter whenever flat) and the baseline now uses spot (no funding). Strategy
  numbers in those runs are unaffected.
