# Tycho: one paper bot, evidence-bound research

**PRE-EXPERIMENTAL / EVIDENCE-BOUND — NO_ECONOMIC_CLAIM.**
Main repository: `ddawnlll/tycho-trading-bot`. NautilusTrader owns orders, fills,
protective stops, positions and balances. Binance supplies public USDT-M market
data. No API keys are needed. Exchange execution (including real money and demo
account execution) is disabled at the programmatic boundary and CLI.

The two existing experts, `trend_long` and `xs_momentum`, are authorized only for
paper observation by the owner. This is not a scientific admission or profitability
claim. No new expert can enter the paper allowlist through research output.

## Run on Windows

Python 3.12 or 3.13 is required. From this repository:

```powershell
uv venv -p 3.12 .venv
uv pip install -p .venv/Scripts/python.exe -e '.[test]'
.venv/Scripts/python.exe -m scripts.nautilus_live
```

`python -m scripts.live` is an alias to the same Nautilus bot. `--mode sandbox`
is an alias for `--mode paper`. There is no live/testnet execution flag.
Use `--run-seconds 155` for a bounded connection/reporting smoke test. Stop a
foreground run with Ctrl+C. One OS lock prevents two local paper instances.

```powershell
.venv/Scripts/python.exe -m scripts.paper_status
Get-Content journal/nautilus/latest.json
New-Item -ItemType File live_state/PAUSE  # block new entries; continue managing exits/stops
Remove-Item -LiteralPath live_state/PAUSE # allow entries again unless drawdown-halted
```

Decisions use complete, synchronized, validated closed 4h bars, sufficient warmup,
fresh streamed prices, fixed expert manifests and a 30-minute decision deadline.
An old bar on startup produces an observation and an explicit ABSTAIN, never a
catch-up order. Signals, market state, entry/exit decisions, fills, ownership,
funding and errors are recorded in `journal/nautilus/<session>.jsonl`. The latest
account snapshot updates every minute, with net PnL, native realized/unrealized PnL,
current/max drawdown, positions, cost deductions and lifecycle status.

Every process start creates a **new 10,000 USDT simulated session**. Historical
sessions are retained separately; positions/balances are not restored by the
Nautilus sandbox adapter. Do not combine fresh sessions into a continuous equity
curve or assume a restart resumes an old position. `paper_status` marks reports
older than three minutes STALE. The node reconnects its data client; after a
process/machine shutdown, run the command again. This is not an installed Windows
service or a promise of uptime while this computer is off.

## Risk and costs

Defaults: 0.5% risk per trade, 5 positions, 2x gross and 1x per-symbol notional.
Pending entries reserve gross exposure. Native reduce-only stops protect partial
fills; denied orders free reservations. Exit ownership is retained until filled.
A 20% drawdown halt flattens positions and blocks new entries for that session.
Missing data or changed expert code/parameters blocks decisions. Network history
fetches run outside the Nautilus event loop so simulated stops keep processing.

Native instrument fees are charged by Nautilus. Additional adverse slippage of
2bp per fill for BTC/ETH and 5bp for other coins is deducted in reported net equity
and sizing. This is a cost reserve, not a claim that the sandbox filled at that
price. Actual published Binance funding settlements (rate and settlement mark)
are applied to the simulated inventory at settlement time, deduplicated per
symbol/time. Pending settlement retrieval is retried; it is not credited as zero
performance evidence. These models cannot prove real exchange fill quality.

## Five candidates, sequential research, no automatic admission

The fixed queue is `trend_long`, `xs_momentum`, `funding_squeeze`,
`squeeze_breakout`, `range_revert`. The last three stay disabled. The sixth legacy
short-only idea is outside this five-candidate scope. There is no optimizer,
learned router, extra execution engine, factory framework or Kanban dependency in
the supported workflow. Older backtest/ccxt modules remain only for regression
and historical reproducibility; their runner and exchange broker are disabled.

Research needs locally cached public data and instrument filters:

```powershell
.venv/Scripts/python.exe -m scripts.download
Invoke-WebRequest https://fapi.binance.com/fapi/v1/exchangeInfo -OutFile data/exchange_info.json
.venv/Scripts/python.exe -m scripts.research_queue
```

The queue uses NautilusTrader for execution and accounting. It writes native
orders/fills/positions, cost-adjusted equity, a 3x-cost stress curve, 30-day block
bootstrap/sign-null diagnostics and a hash-chained trial log. Data, code and expert
manifests are identified. Existing `runs/` results are never inputs to admission.
Historical reruns always report BLOCKED for approval, even when positive.
The latest diagnostic audit found losses for all five fixed candidates; see the
new `*_candidate_audit/summary.json`. No candidate was activated by those runs.

`research/protocol.json` freezes a future 180-day observation window, five
manifests, implementation hashes and provisional review thresholds. It is created
once by `python -m scripts.freeze_research` and refuses overwrite. To review the
frozen window after it completes and public cache coverage is refreshed:

```powershell
.venv/Scripts/python.exe -m scripts.research_queue --prospective
```

This refuses incomplete time windows, changed implementation/manifests or missing
coverage. Thresholds: at least 180 days and 30 closed trades; positive net return
and 3x-cost return; max drawdown at most 20%; positive 5th-percentile block-bootstrap
mean; sign-null p <= 0.01 for a family of five. These are provisional screening
rules, not proof of alpha. Success only produces AWAITING_OWNER_APPROVAL; it never
changes configuration or starts a strategy. An explicit owner-approved code change
is required to extend the two-expert paper allowlist. Real money remains disabled.

Historical funding uses cached hourly funding rates and hourly open prices as mark
proxies. The fixed 2026 universe has survivorship bias. Bar fills omit queue/impact
uncertainty and 30-day bootstrap blocks may be too short for long holding periods.
An owner review must address these limits before making any stronger claim.
[Archived results](docs/LEGACY_RESULTS.md) are contaminated research history.
[Migration notes](docs/MIGRATION.md) record what was selected from `ddawnlll/tycho`.

## Verify

```powershell
.venv/Scripts/python.exe -m pytest -q tests
.venv/Scripts/python.exe -m pyflakes tycho scripts
```

Tests cover causal signal parity, data rejection, broker disablement, manifests,
costs/funding, report drawdowns, native partial fills/protective stops, pending
exposure, trial tampering, process locks and fail-closed review. Synthetic execution
tests establish mechanics, not economic success.
