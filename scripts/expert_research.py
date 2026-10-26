"""Evaluate each expert alone (shared account, same caps), small IS grid, admit on IS-only criteria.

Usage: python -m scripts.expert_research
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C, runlog
from tycho.engine import metrics
from tycho.ensemble import build_columns, window
from tycho.engine_multi import run_multi
from tycho.experts import CATALOG, Expert
from tycho.marketstate import build_state
from tycho.portfolio import load_universe
from scripts.portfolio_research import IS, OOS

GRIDS = {  # (rule params, requires params)
    "trend_long": ({}, {}),
    "trend_short": (dict(entry_n=[20, 30, 55], exit_n=[10, 20]), dict(max_breadth=[0.3, 0.5])),
    "range_revert": (dict(rsi_lo=[5.0, 10.0, 20.0], exit_n=[5, 10]), {}),
    "squeeze_breakout": (dict(sq_pct=[0.1, 0.2], trail_atr=[2.5, 3.5]), dict(max_vol_pct=[0.35, 1.01])),
    "funding_squeeze": ({}, dict(z=[-1.0, -1.5, -2.0])),
    "xs_momentum": (dict(top_k=[2, 3, 4], exit_rank=[5, 7]), {}),
}
ADMIT = dict(min_is_sharpe=0.5, min_is_trades=30)
KW = dict(max_positions=C.MAX_POSITIONS, max_gross=C.MAX_GROSS, max_lev_pos=C.MAX_LEV_POS, risk=C.RISK_DEFAULT)


def expand(g: dict):
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())] if keys else [{}]


def evaluate(data, ms, experts: list[Expert], start, end, kw=KW):
    cols, sigs, costs, groups = build_columns(data, experts, ms)
    d, sg = window(cols, sigs, start, end)
    res = run_multi(d, sg, costs=costs, groups=groups, **kw)
    return res, metrics(res["equity"], res["trades"])


def main():
    data = load_universe(C.UNIVERSE, C.TIMEFRAME)
    ms = build_state(data, C.REGIME_N)
    rows = []
    for name, (rg, qg) in GRIDS.items():
        base = CATALOG[name]
        for rp in expand(rg):
            for qp in expand(qg):
                ex = Expert(name, base.rule, base.requires, rp, qp)
                res_i, mi = evaluate(data, ms, [ex], *IS)
                res_o, mo = evaluate(data, ms, [ex], *OOS)
                ti = res_i["trades"]
                sym_pos = int((ti.groupby(ti.symbol.str.split(":").str[1]).r.sum() > 0).sum()) if len(ti) else 0
                rows.append(dict(expert=name, params=rp, req=qp,
                                 is_cagr=mi["cagr"], is_dd=mi["max_dd"], is_sharpe=mi["sharpe"], is_trades=mi["n_trades"],
                                 is_avg_r=mi.get("avg_r"), is_syms_pos=sym_pos,
                                 oos_cagr=mo["cagr"], oos_dd=mo["max_dd"], oos_sharpe=mo["sharpe"],
                                 oos_trades=mo["n_trades"], oos_avg_r=mo.get("avg_r")))
                r = rows[-1]
                print(f"{name:17s} {str(rp):40s} {str(qp):22s} IS sh {r['is_sharpe']:5.2f} cagr {r['is_cagr']:6.1%} "
                      f"n {r['is_trades']:4d} | OOS sh {r['oos_sharpe']:5.2f} cagr {r['oos_cagr']:6.1%} "
                      f"dd {r['oos_dd']:6.1%} n {r['oos_trades']:4d}", flush=True)
    g = pd.DataFrame(rows)
    picks = {}
    for name, sub in g.groupby("expert", sort=False):
        best = sub.sort_values("is_sharpe", ascending=False).iloc[0]
        admitted = bool(best.is_sharpe >= ADMIT["min_is_sharpe"] and best.is_cagr > 0
                        and best.is_trades >= ADMIT["min_is_trades"])
        picks[name] = dict(params=best.params, req=best.req, admitted=admitted, n_tried=len(sub),
                           is_sharpe=best.is_sharpe, oos_sharpe=best.oos_sharpe, oos_cagr=best.oos_cagr)
    d = runlog.new_run("expert_research")
    adm = [k for k, v in picks.items() if v["admitted"]]
    headline = f"admitted on IS: {adm}; " + "; ".join(
        f"{k} OOS sh {v['oos_sharpe']:.2f}" for k, v in picks.items())
    runlog.save(d, dict(grids=GRIDS, admit=ADMIT, kw=KW, IS=IS, OOS=OOS),
                dict(headline=headline, picks=picks), {"grid": g})
    for k, v in picks.items():
        print(k, v)
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
