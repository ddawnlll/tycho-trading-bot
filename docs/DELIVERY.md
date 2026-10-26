# Delivery verification — 2026-10-26

Main repo: `ddawnlll/tycho-trading-bot`, branch `codex/paper-expert-evidence`.

- Full local regression run: **46 passed**, 408.53 seconds; no skipped tests.
- Final focused safety/order suite after the last changes: **31 passed**, including
  fresh-bar decision idempotency, PAUSE, stale stream rejection and portable hashes.
- `python -m pyflakes tycho scripts`: passed. `git diff --check`: passed.
- Editable package installation with `uv pip install --no-deps -e .`: passed.
- Bounded Binance public-data/Nautilus sandbox session: connected all 12 symbols,
  emitted account snapshots, calculated both expert plans and explicitly abstained
  from a stale bar. No exchange execution adapter was constructed.
- Final background paper session: `20261026T172820-7e4ceb38`, RUNNING at
  2026-10-26 20:30:21 Europe/Istanbul; balance 10,000 USDT, net PnL 0, drawdown 0,
  open positions 0. This is connectivity/operation evidence, not economic evidence.
- Final five-candidate audit: `runs/20261026T172802_candidate_audit/summary.json`.
  Net returns: trend_long -3.84%, xs_momentum -3.11%, funding_squeeze -1.86%,
  squeeze_breakout -2.68%, range_revert -5.03%. All approval verdicts BLOCKED.
  These are contaminated historical diagnostics, not fresh out-of-sample evidence.
- Trial hash chain: 33 entries verified. Frozen source hashes: 21 files verified.
- Prospective invocation correctly refused the incomplete future window.
  Frozen window: 2026-10-26 23:00 through 2027-04-24 23:00 Europe/Istanbul;
  evaluate only after the final bar closes. An initial newline-dependent protocol
  was superseded before its window started; its bytes and the reason are retained.

No real-money path was enabled. No new expert was activated. The existing two
experts are owner-authorized paper observations only. Process restarts create a
fresh simulated account; the program is not installed as a Windows service.

To read current state: `python -m scripts.paper_status`. Operational JSONL and logs
are local under `journal/`; native synthetic fill/cost tests run in CI without
cached Binance data. Data-dependent historical parity tests require the public
cache and explicitly skip when that cache is absent.
