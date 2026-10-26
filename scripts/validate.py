"""Stress-test the configured expert ensemble: per-year returns, cost stress, bootstrap drawdown.

Usage: python -m scripts.validate [--risk 0.005]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C, runlog
from tycho.engine import metrics
from tycho.engine_multi import run_multi
from tycho.ensemble import build_columns, configured_experts, window
from tycho.marketstate import build_state
from tycho.portfolio import load_universe


def kw(risk):
    return dict(max_positions=C.MAX_POSITIONS, max_gross=C.MAX_GROSS, max_lev_pos=C.MAX_LEV_POS, risk=risk)


def run_window(cols, sigs, costs, groups, start, end, risk):
    d, s = window(cols, sigs, start, end)
    res = run_multi(d, s, costs=costs, groups=groups, **kw(risk))
    return res, metrics(res["equity"], res["trades"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, default=C.RISK_DEFAULT)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    data = load_universe(C.UNIVERSE, C.TIMEFRAME)
    out = {}
    ms = build_state(data, C.REGIME_N)
    cols, sigs, costs, groups = build_columns(data, configured_experts(), ms)
    res, m_full = run_window(cols, sigs, costs, groups, "2020-01-01", "2030-01-01", a.risk)
    eq, tr = res["equity"], res["trades"]
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    ydd = eq.groupby(eq.index.year).apply(lambda e: (e / e.cummax() - 1).min())
    ytab = pd.DataFrame({"return": yearly.values, "max_dd_in_year": ydd.values,
                         "trades": tr.groupby(tr.exit_time.dt.year).size().reindex(ydd.index).fillna(0).values},
                        index=ydd.index)
    monthly = eq.resample("ME").last().pct_change().dropna()
    out["full"] = m_full
    out["share_months_positive"] = float((monthly > 0).mean())
    out["worst_month"] = float(monthly.min())
    out["best_month"] = float(monthly.max())
    # cost stress
    for mult in (2.0, 3.0):
        c2 = {k: type(c)(fee=c.fee * mult, slippage=c.slippage * mult) for k, c in costs.items()}
        _, m = run_window(cols, sigs, c2, groups, "2024-01-01", "2030-01-01", a.risk)
        out[f"oos_costs_x{mult:g}"] = m
    _, out["oos_base"] = run_window(cols, sigs, costs, groups, "2024-01-01", "2030-01-01", a.risk)
    # block bootstrap of MONTHLY returns (3-month blocks keep loss clustering; i.i.d. trades understate risk)
    rng = np.random.default_rng(7)
    mr = monthly.to_numpy()
    blk, dds, yrs = 3, [], []
    for _ in range(a.n_boot):
        starts = rng.integers(0, len(mr) - blk + 1, size=len(mr) // blk + 1)
        path = np.concatenate([mr[i:i + blk] for i in starts])[:len(mr)]
        e = np.cumprod(1 + path)
        dds.append((e / np.maximum.accumulate(e) - 1).min())
        yrs.append(np.prod(1 + path[:12]) - 1)
    out["bootstrap"] = dict(method="3-month block bootstrap of monthly returns",
                            maxdd_p50=float(np.percentile(dds, 50)), maxdd_p95=float(np.percentile(dds, 5)),
                            year_ret_p5=float(np.percentile(yrs, 5)), year_ret_p50=float(np.percentile(yrs, 50)),
                            prob_losing_year=float(np.mean(np.array(yrs) < 0)))
    d = runlog.new_run(f"validate_risk{a.risk}")
    headline = (f"risk {a.risk}: full CAGR {m_full['cagr']:.1%} DD {m_full['max_dd']:.1%} Sharpe {m_full['sharpe']:.2f}; "
                f"OOS base {out['oos_base']['cagr']:.1%}, x3 costs {out['oos_costs_x3']['cagr']:.1%}; "
                f"P(losing year) {out['bootstrap']['prob_losing_year']:.0%}")
    runlog.save(d, vars(a) | {k: getattr(C, k) for k in dir(C) if k.isupper()}, out | {"headline": headline},
                {"yearly": ytab, "monthly": monthly.to_frame("ret"), "trades": tr,
                 "equity_daily": eq.resample("1D").last().to_frame()})
    print(ytab.round(3).to_string())
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["bootstrap"].items()})
    for k in ("oos_base", "oos_costs_x2", "oos_costs_x3"):
        print(k, {x: round(out[k][x], 3) for x in ("cagr", "max_dd", "sharpe", "n_trades")})
    print(f"months positive {out['share_months_positive']:.0%}, worst month {out['worst_month']:.1%}, best {out['best_month']:.1%}")
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
