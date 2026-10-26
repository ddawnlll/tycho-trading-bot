"""Grid-search a strategy in-sample, then evaluate the chosen params ONCE out-of-sample.

Usage: python scripts/research.py --symbol BTCUSDT --tf 4h --strategy trend_pullback
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
from tycho.data import load, resample
from tycho.engine import Costs, Signals, metrics, run
from tycho.strategies import STRATEGIES

IS_END = "2023-12-31 23:59"
OOS_START = "2024-01-01"

GRIDS = {
    "trend_pullback": dict(fast=[9], mid=[21], slow=[50, 100], hi_lookback=[20, 40],
                           max_pb_bars=[5, 8, 12], stop_atr=[0.25, 0.5, 1.0],
                           trail_atr=[2.0, 3.0, 4.0], trail_n=[10, 20], allow_short=[True, False]),
    "donchian": dict(entry_n=[20, 40, 55, 100], exit_n=[10, 20, 40], stop_atr=[1.5, 2.5, 4.0],
                     allow_short=[True, False]),
}


def slice_sig(sig: Signals, mask: np.ndarray) -> Signals:
    return Signals(*(getattr(sig, f)[mask] for f in sig.__dataclass_fields__))


def evaluate(df, sig, mask, risk, lev, costs):
    res = run(df[mask], slice_sig(sig, mask), risk=risk, max_lev=lev, costs=costs)
    return metrics(res["equity"], res["trades"]), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--strategy", default="trend_pullback")
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--lev", type=float, default=3.0)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--min_trades", type=int, default=40)
    a = ap.parse_args()

    df = resample(load(a.symbol), a.tf)
    is_mask = (df.index <= IS_END)
    oos_mask = (df.index >= OOS_START)
    costs = Costs(slippage=a.slippage)
    fn = STRATEGIES[a.strategy]
    grid = GRIDS[a.strategy]
    keys = list(grid)
    rows = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(keys, vals))
        sig = fn(df, **p)
        m, _ = evaluate(df, sig, is_mask, a.risk, a.lev, costs)
        rows.append({**p, **{f"is_{k}": v for k, v in m.items() if k not in ("start", "end")}})
    g = pd.DataFrame(rows)
    ok = g[g.is_n_trades >= a.min_trades].copy()
    # robust pick: highest IS Sharpe among sets also in top half by MAR (avoid DD-heavy outliers)
    ok = ok[ok.is_mar >= ok.is_mar.median()]
    best = ok.sort_values("is_sharpe", ascending=False).iloc[0]
    params = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keys}

    sig = fn(df, **params)
    m_is, _ = evaluate(df, sig, is_mask, a.risk, a.lev, costs)
    m_oos, res_oos = evaluate(df, sig, oos_mask, a.risk, a.lev, costs)
    spot = df.assign(funding=0.0)  # baseline = holding spot, no funding
    bh = STRATEGIES["buy_hold"](spot)
    bh_oos, _ = evaluate(spot, bh, oos_mask, 1.0, 1.0, costs)
    bh_is, _ = evaluate(spot, bh, is_mask, 1.0, 1.0, costs)
    # what share of the whole grid is profitable OOS? (overfit diagnostic, computed AFTER the pick)
    oos_all = []
    for _, r in g.iterrows():
        p = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keys}
        mo, _ = evaluate(df, fn(df, **p), oos_mask, a.risk, a.lev, costs)
        oos_all.append(mo["cagr"])
    g["oos_cagr"] = oos_all
    rank_corr = float(g[["is_sharpe", "oos_cagr"]].corr(method="spearman").iloc[0, 1])

    d = runlog.new_run(f"{a.strategy}_{a.symbol}_{a.tf}")
    headline = (f"{a.strategy} {a.symbol} {a.tf}: OOS CAGR {m_oos['cagr']:.1%} DD {m_oos['max_dd']:.1%} "
                f"Sharpe {m_oos['sharpe']:.2f} vs B&H CAGR {bh_oos['cagr']:.1%} DD {bh_oos['max_dd']:.1%}")
    summary = dict(headline=headline, params=params, n_grid=len(g), is_metrics=m_is, oos_metrics=m_oos,
                   buy_hold_is=bh_is, buy_hold_oos=bh_oos,
                   grid_share_oos_profitable=float((g.oos_cagr > 0).mean()),
                   grid_is_oos_spearman=rank_corr)
    runlog.save(d, vars(a) | {"is_end": IS_END, "oos_start": OOS_START, "grid": grid}, summary,
                {"grid": g, "oos_trades": res_oos["trades"],
                 "oos_equity": res_oos["equity"].resample("1D").last().to_frame()})
    print(headline)
    print("params:", params)
    print("IS :", {k: round(v, 3) if isinstance(v, float) else v for k, v in m_is.items()})
    print("OOS:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m_oos.items()})
    print(f"grid {len(g)} sets | share profitable OOS {summary['grid_share_oos_profitable']:.0%} "
          f"| IS-Sharpe vs OOS-CAGR spearman {rank_corr:.2f}")
    print("saved", d)


if __name__ == "__main__":
    main()
