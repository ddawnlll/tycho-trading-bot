"""Evaluate five fixed candidates sequentially in Nautilus, never activate them.

Existing data is contaminated historical context. Fresh reruns do not make it OOS.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from tycho import config as C
from tycho import nautilus_bt as NB
from tycho.data import resample
from tycho.ensemble import configured_experts, build_columns
from tycho.evidence import CANDIDATES, append_trial, review, return_diagnostics, cost_adjusted_equity
from tycho.reporting import clean
from tycho.safety import canonical_hash, expert_manifest, validate_bars, source_hash
from tycho.locking import process_lock

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--prospective", action="store_true", help="use only the frozen future window; refuse if incomplete")
    a = ap.parse_args()
    if a.days < 60:
        ap.error("need at least 60 days")
    protocol = None
    if a.prospective:
        protocol = json.loads((ROOT / "research" / "protocol.json").read_text(encoding="utf-8"))
        for file, digest in protocol["files"].items():
            if source_hash(ROOT / file) != digest:
                raise SystemExit(f"BLOCKED: frozen implementation changed: {file}; new preregistration required")
        if pd.Timestamp.now(tz="UTC") < pd.Timestamp(protocol["end"]) + pd.Timedelta(C.TIMEFRAME):
            raise SystemExit("BLOCKED: prospective window is not complete; historical evidence is never substituted")
    hourly = {s: pd.read_parquet(ROOT / "data" / f"{s}_1h.parquet") for s in C.UNIVERSE}
    data = {s: resample(df, C.TIMEFRAME) for s, df in hourly.items()}
    end = min(df.index[-1] for df in data.values())
    # Explicit common closed window, never implicitly refresh/download during a trial.
    end = min(end, pd.Timestamp.now(tz="UTC").floor(C.TIMEFRAME) - pd.Timedelta(C.TIMEFRAME))
    start = end - pd.Timedelta(days=a.days)
    if protocol:
        start, frozen_end = pd.Timestamp(protocol["start"]), pd.Timestamp(protocol["end"])
        if end < frozen_end or any(df.index[0] > start - pd.Timedelta(days=800) for df in data.values()):
            raise SystemExit("BLOCKED: cached data does not cover frozen window and warmup; refresh public data")
        end = frozen_end
        a.days = int((end - start).total_seconds() / 86400)
    filt = NB.exchange_filters()
    data = {s: NB.round_to_instrument(df.loc[:end], NB.make_instrument(s, filt[s])) for s, df in data.items()}
    validate_bars({s: df.tail(C.REGIME_WARMUP_BARS if s == "BTCUSDT" else C.WARMUP_BARS)
                   for s, df in data.items()}, end)
    data_hash = canonical_hash({s: canonical_hash(pd.util.hash_pandas_object(df, index=True).astype(str).tolist())
                                for s, df in hourly.items()})
    run = ROOT / "runs" / (pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S") + "_candidate_audit")
    run.mkdir(parents=True, exist_ok=False)
    results = []
    for name in CANDIDATES:
        params, req = next(((p, r) for n, p, r in C.EXPERTS if n == name), ({}, {}))
        expert = configured_experts([(name, params, req)])[0]
        manifest = expert_manifest(expert)
        if protocol and protocol["manifests"].get(name) != manifest:
            raise SystemExit(f"BLOCKED: {name} manifest differs from preregistration")
        log = ROOT / "research" / "trials.jsonl"
        append_trial(log, dict(event="started", expert=name, manifest=manifest, data_hash=data_hash,
                               start=str(start), end=str(end), classification="PROSPECTIVE" if protocol else "HISTORICAL_DIAGNOSTIC"))
        cols, sigs, _, _ = build_columns(data, [expert])
        result = NB.run(data, cols, sigs, start, end)
        base, costs = cost_adjusted_equity(result, hourly)
        stress, stress_costs = cost_adjusted_equity(result, hourly, multiplier=3)
        closed = int(result["positions"].ts_closed.notna().sum()) if not result["positions"].empty else 0
        lower, p = return_diagnostics(base)
        ret, stressed = float(base.iloc[-1] / 10000 - 1), float(stress.iloc[-1] / 10000 - 1)
        dd = float((1 - base / base.cummax().clip(lower=10000)).max())
        verdict = review(fresh=bool(protocol), days=a.days, trades=closed, net_return=ret,
                         stress_return=stressed, max_drawdown=dd, bootstrap_lower=lower, null_p=p)
        row = clean(dict(expert=name, manifest=manifest, data_hash=data_hash, start=str(start), end=str(end),
                         closed_trades=closed, net_return=ret, stress_return=stressed, max_drawdown=dd,
                         bootstrap_lower=lower, null_p=p, costs=costs, stress_costs=stress_costs,
                         protocol_hash=canonical_hash(protocol) if protocol else None,
                         **verdict, economic_verdict="NO_ECONOMIC_CLAIM"))
        append_trial(log, dict(event="completed", **row))
        results.append(row)
        pd.DataFrame({"net": base, "stress_3x": stress}).to_csv(run / f"{name}_equity.csv")
        for field in ("positions", "fills", "orders"):
            result[field].to_csv(run / f"{name}_{field}.csv")
        print(json.dumps(row), flush=True)
    (run / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {run}; no expert activated", flush=True)


if __name__ == "__main__":
    with process_lock(ROOT / "live_state" / "research.lock"):
        main()
