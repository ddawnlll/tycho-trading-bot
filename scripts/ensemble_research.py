"""Combine IS-admitted experts in one shared account. Subset + position cap chosen on IS (MAR);
OOS reported for every combination. Expert priority = list order (trend_long first).

Usage: python -m scripts.ensemble_research [--expert_run runs/<ts>_expert_research]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C, runlog
from tycho.experts import CATALOG, Expert
from tycho.marketstate import build_state
from tycho.portfolio import load_universe
from scripts.expert_research import KW, evaluate
from scripts.portfolio_research import IS, OOS


def latest(tag):
    return sorted(p for p in (runlog.RUNS).glob(f"*_{tag}") if p.is_dir())[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_run", default=None)
    a = ap.parse_args()
    src = Path(a.expert_run) if a.expert_run else latest("expert_research")
    picks = json.loads((src / "summary.json").read_text())["picks"]
    admitted = [k for k in CATALOG if picks[k]["admitted"]]
    experts = {k: Expert(k, CATALOG[k].rule, CATALOG[k].requires, picks[k]["params"], picks[k]["req"]) for k in admitted}
    others = [k for k in admitted if k != "trend_long"]
    data = load_universe(C.UNIVERSE, C.TIMEFRAME)
    ms = build_state(data, C.REGIME_N)
    rows = []
    for r in range(len(others) + 1):
        for sub in itertools.combinations(others, r):
            names = ["trend_long", *sub]
            for maxpos in (5, 8):
                kw = KW | {"max_positions": maxpos}
                exs = [experts[n] for n in names]
                ri, mi = evaluate(data, ms, exs, *IS, kw=kw)
                ro, mo = evaluate(data, ms, exs, *OOS, kw=kw)
                by = ro["trades"].assign(expert=ro["trades"].symbol.str.split(":").str[0]).groupby("expert").r
                rows.append(dict(experts="+".join(names), max_positions=maxpos,
                                 is_cagr=mi["cagr"], is_dd=mi["max_dd"], is_sharpe=mi["sharpe"], is_mar=mi["mar"],
                                 oos_cagr=mo["cagr"], oos_dd=mo["max_dd"], oos_sharpe=mo["sharpe"], oos_mar=mo["mar"],
                                 oos_trades=mo["n_trades"], oos_sumR_by_expert=by.sum().round(1).to_dict()))
                x = rows[-1]
                print(f"{x['experts']:45s} pos{maxpos} IS sh {x['is_sharpe']:.2f} cagr {x['is_cagr']:6.1%} dd {x['is_dd']:6.1%} "
                      f"mar {x['is_mar']:.2f} | OOS sh {x['oos_sharpe']:.2f} cagr {x['oos_cagr']:6.1%} dd {x['oos_dd']:6.1%} "
                      f"R {x['oos_sumR_by_expert']}", flush=True)
    t = pd.DataFrame(rows)
    pick = t.sort_values("is_mar", ascending=False).iloc[0]
    base = t[(t.experts == "trend_long") & (t.max_positions == C.MAX_POSITIONS)].iloc[0]
    d = runlog.new_run("ensemble_research")
    headline = (f"IS-pick {pick.experts} pos{pick.max_positions}: OOS CAGR {pick.oos_cagr:.1%} DD {pick.oos_dd:.1%} "
                f"Sharpe {pick.oos_sharpe:.2f} vs trend_long alone OOS CAGR {base.oos_cagr:.1%} Sharpe {base.oos_sharpe:.2f}")
    runlog.save(d, dict(expert_run=src.name, admitted=admitted, experts={k: vars(v) | {"rule": v.rule.__name__,
                "requires": v.requires.__name__} for k, v in experts.items()}, kw=KW, IS=IS, OOS=OOS),
                dict(headline=headline, pick=pick.to_dict()), {"ensembles": t})
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
