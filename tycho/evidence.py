"""Hash-chained trials and fail-closed research review, adapted from ddawnlll/tycho.

Research never edits the paper allowlist. A reviewable result is not activation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .safety import canonical_hash

CANDIDATES = ("trend_long", "xs_momentum", "funding_squeeze", "squeeze_breakout", "range_revert")


def read_trials(path):
    path = Path(path)
    previous = "GENESIS"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k != "hash"}
            if body.get("prev_hash") != previous or row.get("hash") != canonical_hash(body):
                raise ValueError("Trial log integrity failure")
            previous = row["hash"]
            rows.append(row)
    return rows


def append_trial(path, record):
    if {"hash", "prev_hash"} & record.keys():
        raise ValueError("reserved trial fields")
    rows = read_trials(path)
    body = dict(record, prev_hash=rows[-1]["hash"] if rows else "GENESIS")
    stored = dict(body, hash=canonical_hash(body))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stored, sort_keys=True, allow_nan=False) + "\n")
    return stored


def review(*, fresh, days, trades, net_return, stress_return, max_drawdown,
           bootstrap_lower, null_p, paper_days=0, paper_trades=0, integrity=True, require_paper=False):
    """Explicit provisional policy frozen before prospective observation, family size=5.

    Missing evidence is BLOCKED, malformed evidence INVALID, adverse evidence FAIL.
    No evaluator here accepts old run summaries or owner approval as performance data.
    """
    numbers = [days, trades, net_return, stress_return, max_drawdown, bootstrap_lower,
               null_p, paper_days, paper_trades]
    if any(x is None for x in numbers):
        return dict(verdict="BLOCKED", reasons=["missing evidence"], approval_ready=False)
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in numbers):
        return dict(verdict="INVALID", reasons=["nonfinite or malformed evidence"], approval_ready=False)
    if not integrity or not 0 <= null_p <= 1 or not 0 <= max_drawdown <= 1 or min(days, trades, paper_days, paper_trades) < 0:
        return dict(verdict="INVALID", reasons=["integrity or domain violation"], approval_ready=False)
    reasons = []
    if not fresh:
        reasons.append("historical/reused window is diagnostic only")
    if days < 180 or trades < 30:
        reasons.append("need >=180 prospective days and >=30 closed trades")
    if require_paper and (paper_days < 30 or paper_trades < 20):
        reasons.append("need >=30 paper days and >=20 paper closed trades under frozen manifest")
    failed = net_return <= 0 or stress_return <= 0 or max_drawdown > 0.20 or bootstrap_lower <= 0 or null_p > 0.01
    verdict = "BLOCKED" if reasons else "FAIL" if failed else "AWAITING_OWNER_APPROVAL"
    if failed:
        reasons.append("net/stress return, drawdown, bootstrap or multiplicity-adjusted null criterion failed")
    return dict(verdict=verdict, reasons=reasons, approval_ready=verdict == "AWAITING_OWNER_APPROVAL")


def return_diagnostics(equity, seed=1729, repetitions=2000):
    """Deterministic circular 30-day block bootstrap and sign null of daily net returns.

    Diagnostics only: block size is a provisional policy, not calibrated significance
    for strategies with holding periods longer than 30 days.
    """
    daily = equity.resample("1D").last().dropna()
    returns = daily.pct_change().dropna().to_numpy()
    if len(returns) < 30:
        return None, None
    rng = np.random.default_rng(seed)
    means = []
    nulls = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(returns), int(np.ceil(len(returns) / 30)))
        index = np.concatenate([(s + np.arange(30)) % len(returns) for s in starts])[:len(returns)]
        means.append(float(returns[index].mean()))
        # Flip complete blocks, preserving dependence within a 30-day holding block.
        signs = np.repeat(rng.choice([-1, 1], len(starts)), 30)[:len(returns)]
        nulls.append(float((returns[index] * signs).mean()))
    return float(np.quantile(means, .05)), float((1 + np.count_nonzero(np.array(nulls) >= returns.mean())) / (repetitions + 1))


def cost_adjusted_equity(result, hourly, multiplier=1):
    """Overlay funding and adverse 2/5bp slippage on Nautilus native account equity.

    This is cost attribution, not another execution engine. Fill quantities, times,
    prices, stops and commissions all come from Nautilus. Funding uses cached public
    hourly funding rates and the hourly open as a mark proxy (not a real mark tape).
    """
    eq = result["equity"].copy()
    debits = pd.Series(0.0, index=eq.index)
    fills = result["fills"]
    if fills.empty:
        return eq, dict(slippage=0.0, funding=0.0, extra_fees=0.0)
    slip = funding = extra_fee = 0.0
    # generate_order_fills_report has one row per order (market orders fill once on bars).
    for _, f in fills.iterrows():
        s = str(f.instrument_id).split("-")[0]
        amount = float(f.filled_qty) * float(f.avg_px)
        charge = amount * ((.0002 if s in {"BTCUSDT", "ETHUSDT"} else .0005) * multiplier
                           + .0005 * (multiplier - 1))
        slip += amount * (.0002 if s in {"BTCUSDT", "ETHUSDT"} else .0005) * multiplier
        extra_fee += amount * .0005 * (multiplier - 1)
        stamp = pd.Timestamp(f.ts_last)
        loc = eq.index.searchsorted(stamp)
        if loc < len(eq):
            debits.iloc[loc] += charge
    # Native position report includes closed snapshots and the final open position.
    for _, p in result["positions"].iterrows():
        s = str(p.instrument_id).split("-")[0]
        h = hourly[s]
        opened = pd.Timestamp(p.ts_opened)
        closed = pd.Timestamp(p.ts_closed) if pd.notna(p.ts_closed) else eq.index[-1] + pd.Timedelta("1ns")
        window = h[(h.index > opened) & (h.index <= closed) & (h.funding != 0)]
        side = 1 if str(p.entry) == "BUY" else -1
        for t, row in window.iterrows():
            # Stress amplifies debits but never amplifies favorable funding credits.
            charge = float(p.peak_qty) * side * float(row.open) * float(row.funding)
            charge *= multiplier if charge > 0 else 1
            funding += charge
            loc = eq.index.searchsorted(t)
            if loc < len(eq):
                debits.iloc[loc] += charge
    return eq - debits.cumsum(), dict(slippage=slip, funding=funding, extra_fees=extra_fee)
