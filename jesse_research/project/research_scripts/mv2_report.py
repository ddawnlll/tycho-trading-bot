"""Build /results/report.json and /results/REPORT.md from stage outputs (no new backtests)."""
import hashlib
import json
from pathlib import Path

RES, DATA, REPO = Path("/results"), Path("/data"), Path("/repo")


def load(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def verify_ledger(path):
    prev, n = "GENESIS", 0
    for line in path.read_text().strip().split("\n"):
        row = json.loads(line)
        h = row.pop("hash")
        if row["prev"] != prev or hashlib.sha256(json.dumps(row, sort_keys=True, default=float).encode()).hexdigest() != h:
            return {"rows": n, "chain_ok": False, "broken_at_row": n + 1}
        prev, n = h, n + 1
    return {"rows": n, "chain_ok": True, "head": prev}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def f2(x):
    return "n/a" if x is None else f"{x:.2f}"


def main():
    pre = load(REPO / "preregistration.json")
    dev = load(RES / "dev_analysis.json", {})
    ml = load(RES / "ml.json", {})
    hold = load(RES / "holdout.json", {})
    man = load(DATA / "manifest.json", {})
    uni = load(DATA / "universe.json", {})
    env = load(RES / "environment.json", {})
    failures = load(REPO / "failures.json", [])
    led = verify_ledger(REPO / "trials.jsonl")
    trials = [json.loads(x) for x in (REPO / "trials.jsonl").read_text().strip().split("\n")]
    runs = []
    for f in sorted((RES / "cache").glob("*/*/*/*.json")):
        r = json.loads(f.read_text())
        runs.append({"market": r["market"], "chunk": r["chunk"], "symbol": r["symbol"], "hyp": r["hyp"],
                     "params": r["params"], "variant": r["variant"], "slippage": r["slippage"], "trades": len(r["trades"]),
                     "net_pnl_usd": round(sum(t["pnl"] - t["funding"] for t in r["trades"]), 2),
                     "fees_usd": round(sum(t["fee"] for t in r["trades"]), 2), "funding_usd": round(r["funding_total"], 2),
                     "final_equity_usd": round(r["equity"][-1][1], 2), "filled_minutes": r["filled_minutes"]})
    with (RES / "jesse_runs.csv").open("w") as fh:
        cols = list(runs[0]) if runs else []
        fh.write(",".join(cols) + "\n")
        for r in runs:
            fh.write(",".join(json.dumps(r[c]).replace(",", ";") if isinstance(r[c], dict) else str(r[c]) for c in cols) + "\n")
    status = [
        ["Binance USDT-M perps 2020-01..2026-09", "INSPECTED (Tycho v1 and this study's walk-forward)"],
        ["Binance spot 2017-08..2019-12, Tycho holdout-B coins", "INSPECTED (Tycho v1)"],
        ["Binance spot 2018-H2..2019-H2, other coins (this holdout)", "INSPECTED since this study (read once for H2+LR)" if hold.get("survivors") else "UNTOUCHED"],
        ["Binance perps from 2026-10-26 20:00 UTC (Tycho frozen window)", "UNTOUCHED (never requested)"],
    ]
    report = {"name": pre["name"], "jesse_runs": runs, "data_status": status, "economic_verdict": "NO_ECONOMIC_CLAIM", "environment": env,
              "preregistration_sha256": hashlib.sha256((REPO / "preregistration.json").read_bytes()).hexdigest(),
              "data_manifest": man, "universe": uni, "dev_analysis": dev, "ml": ml, "holdout": hold,
              "ledger": led, "trials": trials, "failures": failures}
    (RES / "report.json").write_text(json.dumps(report, indent=1, default=float))

    L = []
    w = L.append
    w(f"# {pre['name']}: Jesse momentum research report\n")
    w("**NO_ECONOMIC_CLAIM. Research and simulation only.** Jesse is the research engine; Tycho/NautilusTrader is untouched "
      "and remains the execution system. Nothing was activated, deployed or connected to an exchange account.\n")
    w("## Bottom line\n")
    w("No momentum candidate survived. The three preregistered rules (H1 pullback, H2 breakout, H3 relative strength) were rejected "
      "in walk-forward on Binance USDT-M perpetuals 2021-H1..2026-H2 (H2 narrowly: every check passed except the block bootstrap, "
      "whose 5th percentile was -2.7e-7/day). The logistic-regression filter on H2 passed every walk-forward criterion on the 9 "
      "gated chunks (Sharpe 0.82, CAGR 2.0%, DD 2.5%, null p 0.001), but that is in-sample-inspected data and one of five candidates; "
      "on the untouched holdout it lost money (return -0.1%, Sharpe -0.03, 57 trades) and was rejected. The holdout's power is low "
      "(3 half-years, 3-31 coins, spot 2018-19), so this is not proof that the H2+LR effect is absent either. Nothing is admitted.\n")
    w("## Verdicts\n")
    w("| hypothesis | walk-forward verdict | OOS CAGR | Sharpe | max DD | trades | 3x-cost CAGR | null p | ML | holdout |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for h, v in dev.get("walk_forward", {}).items():
        o = v["oos"]
        hv = hold.get("results", {}).get(h, {}).get("verdict", "not read (rule rejected)")
        w(f"| {h} rule | **{v['verdict']}** | {pct(o['cagr'])} | {f2(o['sharpe'])} | {pct(o['max_dd'])} | {o['trades']} | "
          f"{pct(v['stress_3x']['cagr'])} | {v['null']['p']:.3f} | {ml.get(h, {}).get('verdict', 'n/a')} | {hv} |")
        mv = ml.get(h, {})
        if "ml" in mv:
            fw = mv.get("full_walk_forward", {})
            o = mv["ml"]
            hv = hold.get("results", {}).get(h + "+LR", {}).get("verdict", "not read")
            w(f"| {h} + LR filter | **{mv['verdict']}** | {pct(o['cagr'])} | {f2(o['sharpe'])} | {pct(o['max_dd'])} | {o['trades']} | "
              f"{pct(mv['ml_stress_3x']['cagr'])} | {fw['null']['p']:.3f} | (is the ML variant) | **{hv}** |" if fw else
              f"| {h} + LR filter | **{mv['verdict']}** | {pct(o['cagr'])} | {f2(o['sharpe'])} | {pct(o['max_dd'])} | {o['trades']} | "
              f"{pct(mv['ml_stress_3x']['cagr'])} | n/a | (is the ML variant) | {hv} |")
    w("")
    w("LR rows cover only the walk-forward chunks whose training fold passed the independence gate "
      "(see the ML section), compared with the rule on the same chunks.\n")
    if hold:
        w(f"Holdout: {hold.get('note', 'read once for ' + ', '.join(hold.get('survivors', [])))}\n")

    w("## Setup and isolation\n")
    for k, v in env.items():
        w(f"- {k}: {v}")
    w("")
    w("## Data\n")
    w(f"- Source: {man.get('source')} public monthly archives, sha256 CHECKSUM verified per file; hard cutoff month "
      f"{man.get('hard_cutoff_month')} (Tycho frozen window 2026-10-26 20:00 UTC onward never read).")
    w(f"- Symbols kept: {man.get('symbols_kept')}; archived symbols whose data ends before the period end (delisted/renamed): "
      f"{man.get('ended_before_period_end')}.")
    w(f"- 1m files: {man.get('minute_files')}; absent monthly files: {man.get('minute_files_missing')}; "
      f"missing minutes inside files (flat-filled): {man.get('missing_minutes_total')} (all on spot; perp files had none); "
      f"OHLC anomalies: {man.get('ohlc_anomalies_total')}; absent funding files: {man.get('funding_files_missing')}.")
    w("- Survivorship: the S3 archive listing includes delisted symbols and membership is fixed point-in-time at each half-year, "
      "so delisted coins (e.g. LUNA 2022-H1) are traded until their last archived minute. Residual biases: archives start 2020-01 "
      "(2020-H1 empty), symbols absent from the 2026-10-26 exchangeInfo snapshot are assumed crypto, and positions in a delisted "
      "coin close at the last archived 1m close rather than Binance's settlement price.\n")
    w("Universe (top 15 by prior 60-day median daily quote volume):\n")
    for m, chs in uni.items():
        for c in chs:
            w(f"- {m} {c['chunk']} (eligible {c['eligible']}): {' '.join(x['symbol'] for x in c['members']) or '(none)'}")
    w("")
    if dev:
        w("## Baselines (stitched OOS " + f"{dev['oos_chunks'][0]}..{dev['oos_chunks'][-1]})\n")
        w("| baseline | CAGR | Sharpe | max DD | by year |")
        w("|---|---|---|---|---|")
        for k, m in dev["baselines"].items():
            w(f"| {k} | {pct(m['cagr'])} | {f2(m['sharpe'])} | {pct(m['max_dd'])} | "
              + ", ".join(f"{y}: {pct(r)}" for y, r in m.get("by_year", {}).items()) + " |")
        w("")
        w("## Every grid cell at fixed parameters (same OOS period; base / 3x cost / delayed entry)\n")
        w("| cell | CAGR | Sharpe | max DD | trades | win rate | funding USD | 3x CAGR | delay CAGR |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k, row in dev["cells"].items():
            b = row["base"]
            w(f"| {k} | {pct(b['cagr'])} | {f2(b['sharpe'])} | {pct(b['max_dd'])} | {b['trades']} | {pct(b['win_rate'])} | "
              f"{b['funding_usd']:.0f} | {pct(row['stress']['cagr'])} | {pct(row['delay']['cagr'])} |")
        w("")
        w("## Walk-forward detail\n")
        for h, v in dev["walk_forward"].items():
            w(f"### {h}: {v['verdict']}\n")
            w("Selected cell per test half-year (expanding window, train Sharpe / trades): "
              + "; ".join(f"{c}: {p['params']} ({f2(p['train_sharpe'])}/{p['train_trades']})" for c, p in v["picks"].items()))
            o = v["oos"]
            w(f"\n- OOS net: CAGR {pct(o['cagr'])}, total {pct(o['net_return'])}, Sharpe {f2(o['sharpe'])}, max DD {pct(o['max_dd'])}, "
              f"trades {o['trades']}, win rate {pct(o['win_rate'])}, fees {o['fees_usd']:.0f} USD, funding {o['funding_usd']:.0f} USD "
              f"(portfolio of 15 x 10,000 USD slots)")
            w("- By year: " + ", ".join(f"{y}: {pct(r)}" for y, r in o.get("by_year", {}).items()))
            w(f"- 3x cost CAGR {pct(v['stress_3x']['cagr'])}; delayed entry CAGR {pct(v['delayed_entry']['cagr'])}; "
              f"without best symbol ({v['drop_best_symbol']['symbol']}) CAGR {pct(v['drop_best_symbol']['cagr'])}; "
              f"positive cells {v['positive_cells']}/4")
            w(f"- Block bootstrap 5th pct of mean daily return: {v['bootstrap_q05_daily']:.6f}")
            w(f"- Null: {json.dumps(v['null'])}")
            w("- Checks: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v["checks"].items()) + "\n")
    if ml:
        w("## Logistic regression vs. non-ML rule\n")
        for h, v in ml.items():
            w(f"### {h}: {v.get('verdict')}\n")
            if "folds" in v:
                w("| test chunk | train trades | distinct entry days | minority | gate |")
                w("|---|---|---|---|---|")
                for c, f in v["folds"].items():
                    w(f"| {c} | {f['train_trades']} | {f['distinct_entry_days']} | {f['minority']} | {'yes' if f['gate'] else 'no'} |")
            if "ml" in v:
                a, b = v["ml"], v["rule_same_chunks"]
                w(f"\nOn the {len(v['chunks'])} gated chunks: LR filter CAGR {pct(a['cagr'])}, Sharpe {f2(a['sharpe'])}, DD {pct(a['max_dd'])}, "
                  f"trades {a['trades']} vs rule CAGR {pct(b['cagr'])}, Sharpe {f2(b['sharpe'])}, DD {pct(b['max_dd'])}, trades {b['trades']}. "
                  f"3x-cost LR CAGR {pct(v['ml_stress_3x']['cagr'])}. Checks: {v['checks']}. {v.get('note', '')}")
            fw = v.get("full_walk_forward")
            if fw:
                w(f"\nFull walk-forward criteria for the LR variant: null {json.dumps(fw['null'])}; fixed cells with their own fold models "
                  + ", ".join(f"{k}: Sharpe {f2(c['sharpe'])} on {len(c['chunks'])} chunks" for k, c in fw["cells"].items())
                  + f"; delayed entry CAGR {pct(fw['delayed_entry']['cagr'])}; without {fw['drop_best_symbol']['symbol']} CAGR "
                  f"{pct(fw['drop_best_symbol']['cagr'])}; checks {fw['checks']} -> **{fw['verdict']}**")
            w("")
    w("## Holdout\n")
    if not hold:
        w("Not evaluated yet.\n")
    elif not hold.get("survivors"):
        w(hold.get("note", "") + " It remains available, untouched, for a future pre-registered test.\n")
    else:
        w("Read exactly once (Binance spot USDT pairs outside Tycho holdout B, 2018-H2..2019-H2, 2018-H1 had no eligible coin; "
          "perp-like costs, assumed 0.01%/8h funding). It is now INSPECTED and cannot serve as an untouched holdout again.\n")
        for h, v in hold["results"].items():
            m = v["holdout"]
            if v.get("model"):
                w(f"- {h}: LR refit on {v['model']['train_trades']} dev trades of the frozen cell before the read; "
                  f"coefficients {[round(c, 3) for c in v['model']['coef']]}, threshold {v['model']['thr']:.3f}")
            w(f"- {h} {v['params']}: return {pct(m['net_return'])}, Sharpe {f2(m['sharpe'])}, DD {pct(m['max_dd'])}, trades {m['trades']}, "
              f"3x cost return {pct(v['stress_3x']['net_return'])}: {v['verdict']} {v['checks']}")
        w("")
    w("## Trial ledger\n")
    w(f"`trials.jsonl`: {led['rows']} rows, hash chain {'verified' if led['chain_ok'] else 'BROKEN'}: preregistration, both baselines, "
      "every grid cell with its base/3x-cost/delay variants, each walk-forward selection, the ML comparisons, the ML full checks, "
      "the holdout parameter freeze and the holdout read. Every individual Jesse backtest "
      f"({len(runs)} runs: market, chunk, symbol, hypothesis, parameters, variant, trades, net PnL, fees, funding) is in "
      "`results/jesse_runs.csv` and `results/report.json`.\n")
    w("## Data status after this study\n")
    w("| dataset | status |")
    w("|---|---|")
    for d, st in status:
        w(f"| {d} | {st} |")
    w("")
    w("## Failures and incidents\n")
    for f in failures:
        w(f"- {f}")
    w("")
    w("## Uncertainties\n")
    for u in pre.get("known_contamination", []) + [
        "Dev data (perps 2020-H2..2026-H2) was already used by Tycho v1 and read by the author: walk-forward numbers are diagnostics, not clean evidence.",
        "Fills are Jesse bar fills at 4h closes and stop prices (gap-aware); queue position, market impact and liquidation engine behaviour are not modelled.",
        "Funding is applied after each Jesse run from actual settlements (spot holdout: assumed 0.01%/8h); sizing inside Jesse does not see funding.",
        "15 equal slots rebalanced each half-year; 1x H3 shorts can lose more than a slot's capital, which is kept as portfolio loss.",
        "Correlated coins: trades and days are not independent; bootstrap blocks of 30 days may be short for multi-week holds.",
        "Indicators use a fixed 330-bar window (Jesse warm-up slicing), so EMA300 is a truncated EMA.",
        "Multiplicity: 3 rules plus 2 LR variants were tested; the null threshold 0.0167 was set for 3 hypotheses.",
        "The LR comparison covers only gated chunks (H2: 2022-H1 and 2023-H1..2026-H2; 2021 and 2022-H2 had too few training trades).",
        "Holdout power is low and its era/instrument differ (2018-19 spot, assumed funding, 1m gaps flat-filled: 47,225 minutes in the H2+LR runs).",
        "Absolute returns are small by construction (1% risk per trade per 1/15 slot); Sharpe, not CAGR, is the comparable statistic.",
    ]:
        w(f"- {u}")
    w("\n## Next steps\n")
    w("- A hypothesis that failed here is closed; any revision needs a new pre-registration (no re-tuning on these results).")
    w("- The only clean evidence left for perpetuals is prospective: after Tycho's frozen window, or a new forward window "
      "registered before it starts, evaluated in paper mode with owner approval.")
    w("- If a future candidate passes walk-forward, run its holdout once and then confirm it in NautilusTrader with in-venue funding "
      "before any paper trial.\n")
    w("Reproduce: see `README.md` in this directory.")
    (RES / "REPORT.md").write_text("\n".join(L) + "\n")
    print("report written", led)


if __name__ == "__main__":
    main()
