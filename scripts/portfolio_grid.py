"""Portfolio-level parameter grid for one strategy. Selection uses IS only; OOS is recorded for
transparency and to measure IS->OOS rank stability (plateau vs. lucky spike).

Usage: python scripts/portfolio_grid.py --strategy donchian
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import runlog
from tycho.portfolio import backtest, load_universe
from scripts.portfolio_research import IS, OOS, UNIVERSE

GRIDS = {
    "donchian": dict(entry_n=[20, 30, 40, 55, 80, 100], exit_n=[10, 15, 20, 30, 40],
                     stop_atr=[1.0, 1.5, 2.0, 3.0], allow_short=[False]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="donchian")
    ap.add_argument("--risk_total", type=float, default=0.005)
    ap.add_argument("--lev", type=float, default=2.0)
    ap.add_argument("--regime", type=int, default=1200)
    a = ap.parse_args()
    data = load_universe(UNIVERSE, "4h")
    sub_risk = a.risk_total * len(data)
    grid = GRIDS[a.strategy]
    rows = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        _, _, mi, _ = backtest(data, a.strategy, p, *IS, sub_risk, a.lev, a.regime or None)
        _, _, mo, per = backtest(data, a.strategy, p, *OOS, sub_risk, a.lev, a.regime or None)
        rows.append({**p, "is_cagr": mi["cagr"], "is_dd": mi["max_dd"], "is_sharpe": mi["sharpe"],
                     "oos_cagr": mo["cagr"], "oos_dd": mo["max_dd"], "oos_sharpe": mo["sharpe"],
                     "oos_trades": mo["n_trades"], "oos_syms_pos": sum(v["ret"] > 0 for v in per.values())})
    g = pd.DataFrame(rows)
    rho = float(g[["is_sharpe", "oos_sharpe"]].corr(method="spearman").iloc[0, 1])
    top = g.sort_values("is_sharpe", ascending=False)
    pick = top.iloc[0]
    d = runlog.new_run(f"portfolio_grid_{a.strategy}_rg{a.regime}")
    headline = (f"{a.strategy} rg{a.regime}: {len(g)} sets, OOS Sharpe>0 in {(g.oos_sharpe > 0).mean():.0%}, "
                f"median OOS Sharpe {g.oos_sharpe.median():.2f}, IS/OOS spearman {rho:.2f}; IS-pick OOS "
                f"Sharpe {pick.oos_sharpe:.2f} CAGR {pick.oos_cagr:.1%} DD {pick.oos_dd:.1%}")
    runlog.save(d, vars(a) | {"grid": grid, "IS": IS, "OOS": OOS},
                dict(headline=headline, is_pick=pick.to_dict(), spearman=rho), {"grid": g})
    pd.set_option("display.width", 200)
    print(top.head(12).round(3).to_string(index=False))
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
