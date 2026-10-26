"""Cross-check tycho.engine_multi against NautilusTrader on identical data and signals.

Stages (each gated): 1) BTC trend_long  2) 12 coins trend_long  3) configured ensemble.
Both engines: same instrument-rounded bars, slippage 0, funding 0, taker fee 0.05%.
Acceptance (architect review): trade sets match (<=1% unmatched), final equity within 0.5%,
max daily equity gap < 0.5%.

Usage: python -m scripts.nautilus_crosscheck [--start 2024-01-01] [--end 2030-01-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C, runlog
from tycho import nautilus_bt as NB
from tycho.engine import metrics
from tycho.engine_multi import run_multi
from tycho.ensemble import build_columns, configured_experts, window
from tycho.marketstate import build_state
from tycho.portfolio import load_universe

BAR = pd.Timedelta(C.TIMEFRAME)


def stage(name, data, experts, start, end, kw):
    ms = build_state(data, C.REGIME_N)
    cols, sigs, costs, groups = build_columns(data, experts, ms, slippage=False)
    d, sg = window(cols, sigs, start, end)
    ty = run_multi(d, sg, costs=costs, groups=groups, **kw)
    sub = {s: data[s] for s in data}
    nt = NB.run(sub, d, sg, start, end, risk=kw["risk"], max_positions=kw["max_positions"],
                max_gross=kw["max_gross"], max_lev_pos=kw["max_lev_pos"])
    t = ty["trades"]
    t = t[t.reason != "eod"]
    ty_set = {(r.symbol.split(":")[1], r.entry_time) for r in t.itertuples()}
    p = nt["positions"]
    nt_set = set()
    if len(p):
        opened = pd.to_datetime(p["ts_opened"], utc=True)
        closed = pd.to_datetime(p["ts_closed"], utc=True)
        for sym, o, c in zip(p["instrument_id"].astype(str), opened, closed):
            if pd.notna(c):
                nt_set.add((sym.split("-PERP")[0], o))  # nautilus fills at close of bar i == open of bar i+1
    matched = ty_set & nt_set
    unmatched = len(ty_set ^ nt_set)
    # per-trade exit time + prices (nautilus stamps fills at bar close = next bar open; a stop hit
    # inside bar j closes at ts(j)+BAR, a market exit decided at close i at ts(i)+BAR = tycho's fill bar)
    ty_by = {(r.symbol.split(":")[1], r.entry_time): r for r in t.itertuples()}
    exit_ok, d_entry, d_exit = 0, [], []
    if len(p):
        for r in p.itertuples():
            if pd.isna(r.ts_closed):
                continue
            k = (str(r.instrument_id).split("-PERP")[0], pd.Timestamp(r.ts_opened, tz="UTC") if pd.Timestamp(r.ts_opened).tzinfo is None else pd.Timestamp(r.ts_opened))
            tr = ty_by.get(k)
            if tr is None:
                continue
            closed = pd.Timestamp(r.ts_closed)
            closed = closed.tz_localize("UTC") if closed.tzinfo is None else closed
            exit_ok += tr.exit_time in (closed - BAR, closed)
            d_entry.append(abs(float(r.avg_px_open) / tr.entry - 1))
            d_exit.append(abs(float(r.avg_px_close) / tr.exit - 1))
    eq_ty = ty["equity"]
    eq_nt = nt["equity"].copy()
    eq_nt.index = eq_nt.index - BAR  # nautilus logs at bar close; tycho indexes by bar open
    both = pd.concat([eq_ty.rename("tycho"), eq_nt.rename("nautilus")], axis=1).dropna()
    daily = both.resample("1D").last().dropna()
    gap = float((daily.nautilus / daily.tycho - 1).abs().max()) if len(daily) else float("nan")
    final_gap = float(both.nautilus.iloc[-1] / both.tycho.iloc[-1] - 1) if len(both) else float("nan")
    m_ty = metrics(eq_ty, ty["trades"])
    m_nt = metrics(eq_nt, pd.DataFrame())
    res = dict(stage=name, tycho_trades=len(ty_set), nautilus_trades=len(nt_set), matched=len(matched),
               exit_time_matched=exit_ok,
               entry_px_diff_median=float(np.median(d_entry)) if d_entry else float("nan"),
               entry_px_diff_p99=float(np.quantile(d_entry, 0.99)) if d_entry else float("nan"),
               exit_px_diff_median=float(np.median(d_exit)) if d_exit else float("nan"),
               exit_px_diff_p99=float(np.quantile(d_exit, 0.99)) if d_exit else float("nan"),
               unmatched_share=unmatched / max(len(ty_set | nt_set), 1), final_equity_gap=final_gap,
               max_daily_equity_gap=gap, tycho_cagr=m_ty["cagr"], nautilus_cagr=m_nt["cagr"],
               tycho_dd=m_ty["max_dd"], nautilus_dd=m_nt["max_dd"], tycho_sharpe=m_ty["sharpe"],
               nautilus_sharpe=m_nt["sharpe"])
    res["pass"] = bool(res["unmatched_share"] <= 0.01 and exit_ok >= 0.99 * len(matched)
                       and abs(final_gap) < 0.005 and gap < 0.005)
    return res, ty, nt, sorted(ty_set - nt_set)[:10], sorted(nt_set - ty_set)[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2030-01-01")
    a = ap.parse_args()
    full = load_universe(C.UNIVERSE, C.TIMEFRAME)
    filt = NB.exchange_filters()
    data = {s: NB.round_to_instrument(df.assign(funding=0.0), NB.make_instrument(s, filt[s])) for s, df in full.items()}
    kw = dict(max_positions=C.MAX_POSITIONS, max_gross=C.MAX_GROSS, max_lev_pos=C.MAX_LEV_POS, risk=C.RISK_DEFAULT)
    trend = [e for e in configured_experts() if e.name == "trend_long"]
    stages = [("1_btc_trend", {"BTCUSDT": data["BTCUSDT"]}, trend),
              ("2_universe_trend", data, trend),
              ("3_ensemble", data, configured_experts())]
    out, tables = [], {}
    for name, dd, ex in stages:
        res, ty, nt, only_ty, only_nt = stage(name, dd, ex, a.start, a.end, kw)
        res["only_tycho_sample"], res["only_nautilus_sample"] = [str(x) for x in only_ty], [str(x) for x in only_nt]
        out.append(res)
        tables[f"{name}_nautilus_positions"] = nt["positions"]
        tables[f"{name}_tycho_trades"] = ty["trades"]
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        if not res["pass"]:
            print(f"stage {name} FAILED acceptance; stopping")
            break
    d = runlog.new_run("nautilus_crosscheck")
    headline = " | ".join(f"{r['stage']}: {'PASS' if r['pass'] else 'FAIL'} trades {r['matched']}/{r['tycho_trades']} "
                          f"final eq gap {r['final_equity_gap']:.3%}" for r in out)
    runlog.save(d, vars(a) | {"kw": kw, "experts": C.EXPERTS}, dict(headline=headline, stages=out), tables)
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
