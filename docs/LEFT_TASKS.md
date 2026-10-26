# Left tasks — momentum candidates v1 (branch `research/momentum-edge`)

State on 2026-10-27: protocol, walk-forward, dev Nautilus confirmation and the untouched holdout are
done and recorded (`docs/MOMENTUM_V1_RESULTS.md`, `research/momentum_trials.jsonl`). **No candidate
passed; nothing was activated.** The paper bot in `D:\tradebot-final` (main) keeps running untouched;
real money stays disabled. None of the tasks below can change a recorded verdict; new hypotheses need a
new pre-registration.

All data needed for tasks 1–4 is already cached in `data/holdout/` (~200 MB, gitignored; recreate with
`python -m scripts.download_holdout`). No new download is required for them.

## 1. Post-hoc holdout diagnostics (POST_HOC, no verdict change)
Add a `diagnostics` stage to `scripts/momentum_research.py` that appends one ledger row with
`classification="POST_HOC_DIAGNOSTIC_NO_VERDICT_CHANGE"` and saves a run:
- [ ] Re-run the five pre-registered final configs on holdout A, save trades, and count positions still
      open (`reason == "eod"`) in coins whose data ended before 2026-09-30 (the 20 series listed in the
      results doc). Report their R and whether they blocked slots.
- [ ] Re-run leader_breakout and dual_momentum with BTC excluded from the momentum rank and the 200-bar
      history rule applied to the rank (needs optional `rank_exclude` / `strict_rank` arguments in
      `tycho/momentum.pit_cross_section`; keep defaults = current behaviour so recorded runs reproduce).
- [ ] Move the context baselines (BTC buy & hold, point-in-time liquid equal-weight alt basket) into
      that stage so the numbers in the holdout table are reproducible from committed code.

## 2. Engine correctness (frozen files — needs a new freeze/pre-registration to change)
- [ ] `tycho/engine_multi.py`: force-close a position at the last bar of a coin whose data ends
      (delisting), instead of marking it at the last close until the run ends.
- [ ] Decide one funding convention for both engines (Binance settles at 00/08/16 UTC = 4h bar opens:
      positions held at the bar open should pay). `tycho/nautilus_funding.FundingModule` currently
      charges positions still open at bar close.

## 3. Tests
- [ ] `tests/test_nautilus_funding.py`: add a mid-bar stop inside a funding bar (documents the
      convention difference until task 2 is done) and avoid comparing against `equity[-2]` with a loose
      tolerance.
- [ ] Unit test for the diagnostics stage on synthetic data (stuck delisted position is detected).

## 4. NautilusTrader on holdout A
- [ ] Confirm holdout A in NautilusTrader. Two attempts for tsmom were stopped by the memory watchdog
      (1.13 GB, 1.02 GB free) while the paper bot was running. Options: run when the paper bot is
      stopped, on a machine with more RAM, or stream bars per instrument instead of building all Bar
      objects up front. Until then holdout verdicts rest on tycho (parity ≤ 1% measured on dev only).

## 5. Data / survivorship (new experiment, needs pre-registration)
- [ ] Download the perps skipped by the `>= 12 monthly files` rule (~300 short-lived symbols, ~10–20 MB,
      ~10–15 min, archives ≤ 2026-09 only) and report holdout A with them included as a separate,
      pre-registered robustness run.
- [ ] Report archive gaps: the downloader silently drops failed months (48 coins have gaps > 1 day,
      many around 2022-03-01).

## 6. Prospective evidence (only remaining clean data)
- [ ] Keep the paper bot running through the frozen window (2026-10-26 20:00 → 2027-04-24 20:00 UTC);
      evaluate only after it closes with `scripts/research_queue.py --prospective`.
- [ ] Any new momentum hypothesis: write a new `research/momentum_protocol_v2.json`, commit it before
      the first backtest, and evaluate it prospectively in paper mode with owner approval.

## 7. Integration
- [ ] Review and merge `research/momentum-edge` into `main` via a pull request (no files in the frozen
      list of `research/protocol.json` were changed; the running paper bot is unaffected until restarted).
