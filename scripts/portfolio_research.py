"""Fixed-rule variants across the whole universe. Pick by IN-SAMPLE Sharpe only; OOS shown for all.

Usage: python scripts/portfolio_research.py [--risk_total 0.005] [--lev 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import runlog
from tycho.portfolio import backtest, load_universe

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"]
IS = ("2020-01-01", "2023-12-31 23:59")
OOS = ("2024-01-01", "2030-01-01")

VARIANTS = {
    "tp_default": ("trend_pullback", {}),
    "tp_btc_is_pick": ("trend_pullback", dict(slow=100, hi_lookback=40, max_pb_bars=8, stop_atr=1.0,
                                              trail_atr=3.0, trail_n=20, allow_short=False)),
    "tp_long": ("trend_pullback", dict(allow_short=False)),
    "donch_55_20": ("donchian", dict(entry_n=55, exit_n=20, stop_atr=2.5)),
    "donch_55_20_long": ("donchian", dict(entry_n=55, exit_n=20, stop_atr=2.5, allow_short=False)),
    "donch_btc_is_pick": ("donchian", dict(entry_n=40, exit_n=20, stop_atr=1.5, allow_short=False)),
    "ma_50_200": ("ma_trend", {}),
    "ma_50_200_long": ("ma_trend", dict(allow_short=False)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk_total", type=float, default=0.005, help="risk per trade as share of TOTAL equity")
    ap.add_argument("--lev", type=float, default=2.0, help="max notional per sub-account / its equity")
    ap.add_argument("--regimes", default="0,1200", help="BTC EMA lengths for regime gate (0 = off)")
    a = ap.parse_args()
    data = load_universe(UNIVERSE, "4h")
    sub_risk = a.risk_total * len(data)
    rows, eqs = [], {}
    for name, (strat, p) in VARIANTS.items():
        for rg in [int(x) for x in a.regimes.split(",")]:
            key = f"{name}|rg{rg}"
            _, _, m_is, _ = backtest(data, strat, p, *IS, sub_risk, a.lev, rg or None)
            eq, tr, m_oos, per = backtest(data, strat, p, *OOS, sub_risk, a.lev, rg or None)
            eqs[key] = eq.resample("1D").last()
            rows.append(dict(variant=key, is_cagr=m_is["cagr"], is_dd=m_is["max_dd"], is_sharpe=m_is["sharpe"],
                             is_trades=m_is["n_trades"], oos_cagr=m_oos["cagr"], oos_dd=m_oos["max_dd"],
                             oos_sharpe=m_oos["sharpe"], oos_trades=m_oos["n_trades"],
                             oos_win=m_oos.get("win_rate"), oos_avg_r=m_oos.get("avg_r"),
                             oos_syms_profitable=sum(v["ret"] > 0 for v in per.values()), n_syms=len(per)))
            print(f"{key:28s} IS cagr {m_is['cagr']:7.1%} dd {m_is['max_dd']:7.1%} sh {m_is['sharpe']:5.2f} | "
                  f"OOS cagr {m_oos['cagr']:7.1%} dd {m_oos['max_dd']:7.1%} sh {m_oos['sharpe']:5.2f} "
                  f"n {m_oos['n_trades']:4d} syms+ {rows[-1]['oos_syms_profitable']}/{len(per)}", flush=True)
    t = pd.DataFrame(rows).set_index("variant")
    pick = t.is_sharpe.idxmax()
    d = runlog.new_run("portfolio_research")
    headline = (f"IS-pick {pick}: OOS CAGR {t.loc[pick, 'oos_cagr']:.1%} DD {t.loc[pick, 'oos_dd']:.1%} "
                f"Sharpe {t.loc[pick, 'oos_sharpe']:.2f}")
    runlog.save(d, vars(a) | {"universe": list(data), "IS": IS, "OOS": OOS, "variants": VARIANTS,
                              "sub_risk": sub_risk},
                dict(headline=headline, is_pick=pick, table=t.to_dict(orient="index")),
                {"variants": t, "oos_equity_daily": pd.DataFrame(eqs)})
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
