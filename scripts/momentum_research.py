"""Momentum candidates v1 (research/momentum_protocol.json): walk-forward, untouched holdouts,
NautilusTrader confirmation. Every trial is appended to research/momentum_trials.jsonl.

  python -m scripts.momentum_research wf        # dev-universe walk-forward + final params
  python -m scripts.momentum_research holdout   # refuses unless the protocol is committed unchanged
  python -m scripts.momentum_research nautilus  # re-run final params in NautilusTrader
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C, runlog
from tycho.engine import metrics
from tycho.engine_multi import run_multi
from tycho.evidence import append_trial, return_diagnostics
from tycho.marketstate import build_state
from tycho.momentum import GRIDS, build, make_expert, window
from tycho.portfolio import load_universe
from tycho.safety import canonical_hash

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "research" / "momentum_protocol.json"
LEDGER = ROOT / "research" / "momentum_trials.jsonl"
DEV_START, DEV_END = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2026-10-26 12:00", tz="UTC")
FROZEN_START = pd.Timestamp("2026-10-26 20:00", tz="UTC")
KW = dict(risk=0.005, max_positions=5, max_gross=2.0, max_lev_pos=1.0)
WF_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
CANDIDATES = ["leader_breakout", "tsmom", "dual_momentum", "vol_expansion", "trend_long"]


def guard(data: dict[str, pd.DataFrame]):
    for s, df in data.items():
        if len(df) and df.index[-1] >= FROZEN_START:
            raise SystemExit(f"BLOCKED: {s} reaches into the frozen prospective window")


def daily(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().dropna().pct_change().dropna()


def stats(rets: pd.Series, n_trades: int) -> dict:
    eq = (1 + rets).cumprod()
    years = max(len(rets) / 365.25, 1e-9)
    sh = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0
    return dict(cagr=float(eq.iloc[-1] ** (1 / years) - 1), total=float(eq.iloc[-1] - 1),
                max_dd=float((1 - eq / eq.cummax()).max()), sharpe=sh, trades=int(n_trades), days=len(rets))


def run_cell(data, ms, name, params, req, start, end, tradable=None, cost_mult=1.0, slip_alt=0.0005):
    ex = make_expert(name, params, req)
    cols, sigs, costs, groups = build(data, [ex], ms, tradable=tradable, cost_mult=cost_mult, slip_alt=slip_alt)
    d, s = window(cols, sigs, start, end)
    res = run_multi(d, s, costs=costs, groups=groups, **KW)
    return res["equity"], res["trades"]


def load_dev():
    data = {s: df[(df.index <= DEV_END)] for s, df in load_universe(C.UNIVERSE, C.TIMEFRAME).items()}
    guard(data)
    return data


def protocol_committed() -> str:
    rel = PROTOCOL.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit("BLOCKED: momentum_protocol.json must be committed and unchanged before holdout runs")
    return canonical_hash(json.loads(PROTOCOL.read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------- stages
def stage_wf():
    data = load_dev()
    ms = build_state(data)
    out, rows = {}, []
    for name in CANDIDATES:
        cells = []
        for params, req in GRIDS[name]:
            eq, tr = run_cell(data, ms, name, params, req, DEV_START, DEV_END)
            eq3, _ = run_cell(data, ms, name, params, req, DEV_START, DEV_END, cost_mult=3.0)
            cells.append(dict(params=params, req=req, rets=daily(eq), rets3=daily(eq3), trades=tr))
        oos, oos3, n, picks = [], [], 0, {}
        for y in WF_YEARS:
            y0, y1 = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
            train = [c["rets"][c["rets"].index < y0] for c in cells]
            score = [float(r.mean() / r.std()) if r.std() > 0 else -np.inf for r in train]
            best = cells[int(np.argmax(score))]
            picks[y] = dict(params=best["params"], req=best["req"])
            oos.append(best["rets"][(best["rets"].index >= y0) & (best["rets"].index < y1)])
            oos3.append(best["rets3"][(best["rets3"].index >= y0) & (best["rets3"].index < y1)])
            t = best["trades"]
            n += int(((t.entry_time >= y0) & (t.entry_time < y1)).sum()) if len(t) else 0
        oos, oos3 = pd.concat(oos), pd.concat(oos3)
        s, s3 = stats(oos, n), stats(oos3, n)
        yearly = {int(y): float((1 + r).prod() - 1) for y, r in oos.groupby(oos.index.year)}
        full = [float(c["rets"].mean() / c["rets"].std()) if c["rets"].std() > 0 else -np.inf for c in cells]
        final = cells[int(np.argmax(full))]
        crit = json.loads(PROTOCOL.read_text(encoding="utf-8"))["pass_criteria"]["walk_forward"]
        ok = (s["cagr"] > crit["net_cagr_gt"] and s["sharpe"] >= crit["sharpe_ge"] and s["max_dd"] <= crit["max_dd_le"]
              and s["trades"] >= crit["trades_ge"] and s3["cagr"] > crit["stress_cagr_gt"])
        out[name] = dict(wf=s, wf_stress=s3, yearly=yearly, picks=picks, wf_pass=bool(ok),
                         final=dict(params=final["params"], req=final["req"]),
                         cells=[dict(params=c["params"], req=c["req"], full_sharpe=f * np.sqrt(365)) for c, f in zip(cells, full)])
        append_trial(LEDGER, dict(event="walk_forward", expert=name, classification="HISTORICAL_WALK_FORWARD",
                                  wf=s, wf_stress=s3, yearly=yearly, wf_pass=bool(ok), final=out[name]["final"],
                                  n_cells=len(cells)))
        rows.append(dict(expert=name, **{f"wf_{k}": v for k, v in s.items()}, stress_cagr=s3["cagr"], wf_pass=ok,
                         **{f"y{y}": v for y, v in yearly.items()}))
        print(json.dumps(rows[-1], default=str), flush=True)
    # baselines over the same OOS span
    span = (pd.Timestamp("2021-01-01", tz="UTC"), DEV_END)
    btc = data["BTCUSDT"]["close"]
    bh = daily(btc[(btc.index >= span[0])])
    basket = pd.concat([daily(df["close"][df.index >= span[0]]) for df in data.values()], axis=1).mean(axis=1).dropna()
    base = dict(btc_buy_hold=stats(bh, 1), equal_weight_basket=stats(basket, len(data)))
    print(json.dumps(base), flush=True)
    d = runlog.new_run("momentum_wf")
    runlog.save(d, dict(protocol=PROTOCOL.name, KW=KW, years=WF_YEARS),
                dict(headline="; ".join(f"{r['expert']} WF sh {r['wf_sharpe']:.2f} cagr {r['wf_cagr']:.1%} "
                                        f"dd {r['wf_max_dd']:.1%} pass {r['wf_pass']}" for r in rows),
                     candidates=out, baselines=base), {"walk_forward": pd.DataFrame(rows)})
    print("saved", d)


def latest_wf() -> dict:
    run = sorted(p for p in runlog.RUNS.glob("*_momentum_wf") if p.is_dir())[-1]
    return json.loads((run / "summary.json").read_text())["candidates"]


def load_holdout_perp():
    files = sorted((ROOT / "data" / "holdout" / "perp").glob("*_4h.parquet"))
    data = {f.name.replace("_4h.parquet", ""): pd.read_parquet(f) for f in files}
    from tycho.momentum import liquid_mask
    data = {s: df for s, df in data.items() if liquid_mask(df).any()}  # never-tradable coins cannot matter
    btc = load_dev()["BTCUSDT"]
    data["BTCUSDT"] = btc[btc.index < pd.Timestamp("2026-10-01", tz="UTC")]
    guard(data)
    return data


def load_holdout_spot():
    files = sorted((ROOT / "data" / "holdout" / "spot").glob("*_4h.parquet"))
    data = {f.name.replace("_4h.parquet", ""): pd.read_parquet(f) for f in files}
    guard(data)
    return data


def evaluate(name, cfg, data, start, end, tradable, slip_alt):
    ms = build_state(data)
    eq, tr = run_cell(data, ms, name, cfg["params"], cfg["req"], start, end, tradable, 1.0, slip_alt)
    eq3, _ = run_cell(data, ms, name, cfg["params"], cfg["req"], start, end, tradable, 3.0, slip_alt)
    m = metrics(eq, tr)
    lower, p = return_diagnostics(eq)
    return dict(cagr=m["cagr"], total=m["total_return"], max_dd=-m["max_dd"], sharpe=m["sharpe"],
                trades=m["n_trades"], win_rate=m.get("win_rate"), avg_r=m.get("avg_r"),
                stress_total=float(eq3.iloc[-1] / eq3.iloc[0] - 1), bootstrap_lower=lower, null_p=p), eq, tr


def stage_holdout():
    phash = protocol_committed()
    wf = latest_wf()
    crit = json.loads(PROTOCOL.read_text(encoding="utf-8"))["pass_criteria"]
    perp = load_holdout_perp()
    spot = load_holdout_spot()
    tradable_perp = [s for s in perp if s != "BTCUSDT"]
    print(f"holdout perps tradable at some point: {len(tradable_perp)}; spot coins: {len(spot)}", flush=True)
    rows, eqs = [], {}
    for name in CANDIDATES:
        cfg = wf[name]["final"]
        hp, eq, tr = evaluate(name, cfg, perp, "2021-01-01", "2026-09-30 20:00", tradable_perp, 0.0010)
        hs, eqs_, _ = evaluate(name, cfg, spot, "2018-01-01", "2019-12-31 20:00", list(spot), 0.0005)
        cp, cs = crit["holdout_perp"], crit["holdout_spot_pre2020"]
        perp_ok = (hp["cagr"] > cp["net_cagr_gt"] and hp["sharpe"] >= cp["sharpe_ge"] and hp["stress_total"] > cp["stress_return_gt"]
                   and (hp["bootstrap_lower"] or -1) > cp["bootstrap_lower_gt"] and (hp["null_p"] if hp["null_p"] is not None else 1) <= cp["null_p_le"])
        spot_ok = hs["total"] > cs["net_return_gt"] and hs["max_dd"] <= cs["max_dd_le"]
        verdict = "PASS" if (wf[name]["wf_pass"] and perp_ok and spot_ok) else "REJECT"
        row = dict(expert=name, final=cfg, wf_pass=wf[name]["wf_pass"], holdout_perp=hp, perp_pass=perp_ok,
                   holdout_spot=hs, spot_pass=spot_ok, verdict=verdict)
        append_trial(LEDGER, dict(event="holdout", classification="UNTOUCHED_HOLDOUT", protocol_hash=phash, **row))
        rows.append(row)
        eqs[f"{name}_perp"], eqs[f"{name}_spot"] = eq.resample("1D").last(), eqs_.resample("1D").last()
        print(json.dumps(row, default=str), flush=True)
    d = runlog.new_run("momentum_holdout")
    runlog.save(d, dict(protocol_hash=phash), dict(headline="; ".join(f"{r['expert']} {r['verdict']}" for r in rows),
                                                  results=rows), {"equity_daily": pd.DataFrame(eqs)})
    print("saved", d)


# ------------------------------------------------------------------------------------------- nautilus
def derived_filters(data: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Tick/step for symbols missing from today's exchangeInfo (delisted): fine enough not to bind."""
    from tycho import nautilus_bt as NB
    known = NB.exchange_filters()
    out = {}
    for s, df in data.items():
        if s in known:
            out[s] = known[s]
            continue
        px = float(df["close"].median())
        tick = 10.0 ** np.floor(np.log10(px) - 5)
        step = 10.0 ** np.floor(np.log10(1.0 / px))  # ~<= 1 USDT per size increment
        out[s] = dict(tick=f"{tick:.12f}".rstrip("0"), step=f"{max(step, 1e-8):.12f}".rstrip("0").rstrip(".") or "1")
    return out


def nautilus_confirm(name, cfg, data, start, end, tradable, slip_alt):
    """Final params in NautilusTrader (fee+slippage as taker fee, funding simulated in the venue) vs
    tycho with the identical cost representation on the same instrument-rounded bars.
    Protocol: disagreement > 1% of final equity => INVALID."""
    from tycho import nautilus_bt as NB
    from tycho import nautilus_funding as NF
    from tycho.engine import Costs
    filt = derived_filters(data)
    fee = {s: 0.0005 + (0.0002 if s in ("BTCUSDT", "ETHUSDT") else slip_alt) for s in data}
    rounded = {s: NB.round_to_instrument(df, NB.make_instrument(s, filt[s])) for s, df in data.items()}
    ms = build_state(rounded)
    ex = make_expert(name, cfg["params"], cfg["req"])
    cols, sigs, _, groups = build(rounded, [ex], ms, tradable=tradable)
    d, sg = window(cols, sigs, start, end)
    nt = NF.run({s: rounded[s] for s in tradable}, d, sg, start, end, filt, fee, **KW)
    costs = {k: Costs(fee=fee[k.split(":")[1]], slippage=0.0) for k in d}
    ty = run_multi(d, sg, costs=costs, groups=groups, **KW)
    eq = nt["equity"]
    gap = float(eq.iloc[-1] / ty["equity"].iloc[-1] - 1)
    m = metrics(eq, pd.DataFrame())
    lower, p = return_diagnostics(eq)
    closed = int(nt["positions"]["ts_closed"].notna().sum()) if len(nt["positions"]) else 0
    return dict(cagr=m["cagr"], total=m["total_return"], max_dd=-m["max_dd"], sharpe=m["sharpe"], closed_trades=closed,
                funding_paid=nt["funding_paid"], bootstrap_lower=lower, null_p=p,
                tycho_total=float(ty["equity"].iloc[-1] / ty["equity"].iloc[0] - 1), final_gap=gap,
                valid=abs(gap) <= 0.01), eq


def stage_nautilus(which: str):
    wf = latest_wf()
    if which == "dev":
        data, start, end, slip_alt = load_dev(), "2021-01-01", str(DEV_END), 0.0005
        tradable = list(data)
    else:
        protocol_committed()
        data, start, end, slip_alt = load_holdout_perp(), "2021-01-01", "2026-09-30 20:00", 0.0010
        tradable = [s for s in data if s != "BTCUSDT"]
    rows, eqs = [], {}
    for name in CANDIDATES:
        res, eq = nautilus_confirm(name, wf[name]["final"], data, start, end, tradable, slip_alt)
        rows.append(dict(expert=name, dataset=which, **res))
        eqs[name] = eq.resample("1D").last()
        append_trial(LEDGER, dict(event="nautilus_confirm", dataset=which, expert=name, final=wf[name]["final"],
                                  classification="HISTORICAL_WALK_FORWARD" if which == "dev" else "UNTOUCHED_HOLDOUT",
                                  **res))
        print(json.dumps(rows[-1], default=str), flush=True)
    d = runlog.new_run(f"momentum_nautilus_{which}")
    runlog.save(d, dict(dataset=which, KW=KW), dict(headline="; ".join(
        f"{r['expert']} NT total {r['total']:.1%} sh {r['sharpe']:.2f} gap {r['final_gap']:.2%}" for r in rows), results=rows),
        {"equity_daily": pd.DataFrame(eqs)})
    print("saved", d)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "wf"
    if stage == "nautilus":
        stage_nautilus(sys.argv[2] if len(sys.argv) > 2 else "dev")
    else:
        {"wf": stage_wf, "holdout": stage_holdout}[stage]()
