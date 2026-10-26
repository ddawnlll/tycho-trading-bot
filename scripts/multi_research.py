"""Shared-account portfolio research. Signal params are FIXED (chosen from the IS plateau of
portfolio_grid); here we choose portfolio constraints on IS (by MAR) and report OOS for all.

Usage: python -m scripts.multi_research
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import runlog
from tycho.engine import Costs, metrics
from tycho.engine_multi import run_multi
from tycho.portfolio import MAJORS, load_universe, regime
from tycho.strategies import STRATEGIES, apply_regime
from scripts.portfolio_research import IS, OOS, UNIVERSE

SIGNAL = ("donchian", dict(entry_n=30, exit_n=30, stop_atr=3.0, allow_short=False))
REGIME_N = 1200


def build(data, strategy=SIGNAL[0], params=SIGNAL[1], regime_n=REGIME_N, slip_mult=1.0):
    bull, bear = regime(data["BTCUSDT"], regime_n)
    sigs, ranks, costs = {}, {}, {}
    for s, df in data.items():
        sig = STRATEGIES[strategy](df, **params)
        sigs[s] = apply_regime(sig, bull.reindex(df.index).fillna(False).to_numpy(bool),
                               bear.reindex(df.index).fillna(False).to_numpy(bool))
        ranks[s] = np.log(df["close"] / df["close"].shift(180)).to_numpy()  # 30-day momentum (leaders)
        costs[s] = Costs(slippage=(0.0002 if s in MAJORS else 0.0005) * slip_mult)
    return sigs, ranks, costs


def window(data, sigs, ranks, start, end):
    d2, s2, r2 = {}, {}, {}
    for s, df in data.items():
        m = (df.index >= start) & (df.index <= end)
        if m.sum() < 50:
            continue
        d2[s] = df[m]
        s2[s] = type(sigs[s])(*(getattr(sigs[s], f)[m] for f in sigs[s].__dataclass_fields__))
        r2[s] = ranks[s][m]
    return d2, s2, r2


def run_window(data, sigs, ranks, costs, start, end, use_rank=True, **kw):
    d, s, r = window(data, sigs, ranks, start, end)
    res = run_multi(d, s, rank=r if use_rank else None, costs=costs, **kw)
    return res, metrics(res["equity"], res["trades"])


def basket_bh(data, start, end):
    eqs = []
    for df in data.values():
        c = df["close"][(df.index >= start) & (df.index <= end)]
        if len(c) > 50:
            eqs.append(c / c.iloc[0])
    idx = pd.DatetimeIndex(sorted(set().union(*[e.index for e in eqs])))
    eq = sum(e.reindex(idx).ffill().bfill() for e in eqs) / len(eqs) * 10_000
    return metrics(eq, pd.DataFrame())


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    data = load_universe(UNIVERSE, "4h")
    sigs, ranks, costs = build(data)
    grid = dict(max_positions=[3, 5, 8, 12], max_gross=[1.0, 2.0, 3.0], risk=[0.005, 0.01], use_rank=[True, False])
    rows = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        kw = dict(max_positions=p["max_positions"], max_gross=p["max_gross"], risk=p["risk"], max_lev_pos=1.0)
        _, mi = run_window(data, sigs, ranks, costs, *IS, use_rank=p["use_rank"], **kw)
        _, mo = run_window(data, sigs, ranks, costs, *OOS, use_rank=p["use_rank"], **kw)
        rows.append({**p, **{f"is_{k}": mi[k] for k in ("cagr", "max_dd", "sharpe", "mar", "n_trades")},
                     **{f"oos_{k}": mo[k] for k in ("cagr", "max_dd", "sharpe", "mar", "n_trades")}})
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    g = pd.DataFrame(rows)
    pick = g[g.is_sharpe > 1.0].sort_values("is_mar", ascending=False).iloc[0]
    bh_is, bh_oos = basket_bh(data, *IS), basket_bh(data, *OOS)
    d = runlog.new_run("multi_research")
    headline = (f"IS-pick pos{int(pick.max_positions)} gross{pick.max_gross} risk{pick.risk} rank{pick.use_rank}: "
                f"OOS CAGR {pick.oos_cagr:.1%} DD {pick.oos_max_dd:.1%} Sharpe {pick.oos_sharpe:.2f} | "
                f"basket B&H OOS CAGR {bh_oos['cagr']:.1%} DD {bh_oos['max_dd']:.1%} Sharpe {bh_oos['sharpe']:.2f}")
    runlog.save(d, {"signal": SIGNAL, "regime_n": REGIME_N, "grid": grid, "IS": IS, "OOS": OOS},
                dict(headline=headline, pick=pick.to_dict(), basket_bh_is=bh_is, basket_bh_oos=bh_oos,
                     share_oos_sharpe_pos=float((g.oos_sharpe > 0).mean()),
                     spearman_is_oos_mar=float(g[["is_mar", "oos_mar"]].corr(method="spearman").iloc[0, 1])),
                {"grid": g})
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
