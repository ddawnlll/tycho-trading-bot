"""XGBoost meta-label filter on the configured strategy's entries, strictly walk-forward.

For test year Y the model is trained only on trades that EXITED before Y-01-01 (labels known then).
It scores each candidate entry bar in Y; entries with p < threshold are dropped. Then the shared-
account engine is re-run with filtered entries and compared to the unfiltered strategy.

Usage: python -m scripts.meta_xgb
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C, runlog
from tycho.engine import metrics
from tycho.engine_multi import run_multi
from tycho.features import bar_features
from tycho.portfolio import load_universe
from scripts.multi_research import build

TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
KW = dict(max_positions=C.MAX_POSITIONS, max_gross=C.MAX_GROSS, max_lev_pos=C.MAX_LEV_POS, risk=C.RISK_DEFAULT)


def main():
    data = load_universe(C.UNIVERSE, C.TIMEFRAME)
    sigs, _, costs = build(data, C.STRATEGY, C.PARAMS, C.REGIME_N)
    feats = {s: bar_features(df, data["BTCUSDT"]) for s, df in data.items()}
    # labels from an unconstrained per-symbol run (every signal taken), so each candidate gets a label
    base = run_multi(data, sigs, costs=costs, max_positions=99, max_gross=1e9, max_lev_pos=1.0, risk=C.RISK_DEFAULT)
    t = base["trades"].copy()
    rows = []
    for r in t.itertuples():
        idx = data[r.symbol].index
        k = idx.get_loc(r.entry_time) - 1  # decision bar
        rows.append(feats[r.symbol].iloc[k].to_dict() | dict(symbol=r.symbol, decision_time=idx[k],
                                                               exit_time=r.exit_time, r=r.r))
    D = pd.DataFrame(rows)
    fcols = [c for c in feats["BTCUSDT"].columns]
    D["y"] = (D.r > 0).astype(int)
    keep = {s: np.ones(len(df), bool) for s, df in data.items()}
    report = []
    for Y in TEST_YEARS:
        y0, y1 = pd.Timestamp(f"{Y}-01-01", tz="UTC"), pd.Timestamp(f"{Y + 1}-01-01", tz="UTC")
        tr = D[D.exit_time < y0]
        te = D[(D.decision_time >= y0) & (D.decision_time < y1)]
        if len(tr) < 100 or len(te) == 0:
            continue
        model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, eval_metric="logloss")
        model.fit(tr[fcols], tr.y)
        thr = float(np.median(model.predict_proba(tr[fcols])[:, 1]))  # drop the bottom half (IS-calibrated)
        for s, df in data.items():
            m = (df.index >= y0) & (df.index < y1) & sigs[s].long_entry
            if m.any():
                p = model.predict_proba(feats[s].loc[m, fcols])[:, 1]
                keep[s][np.where(m)[0]] = p >= thr
        p_te = model.predict_proba(te[fcols])[:, 1]
        taken = te[p_te >= thr]
        report.append(dict(year=Y, n_train=len(tr), n_test=len(te), n_taken=len(taken),
                           sumR_all=te.r.sum(), sumR_taken=taken.r.sum(), sumR_dropped=te.r.sum() - taken.r.sum(),
                           win_all=(te.r > 0).mean(), win_taken=(taken.r > 0).mean() if len(taken) else np.nan))
    rep = pd.DataFrame(report)
    filt = {s: type(sig)(sig.long_entry & keep[s], *(getattr(sig, f) for f in list(sig.__dataclass_fields__)[1:]))
            for s, sig in sigs.items()}
    res = {}
    for name, sg in (("base", sigs), ("xgb_filtered", filt)):
        out = run_multi(data, sg, costs=costs, **KW)
        e = out["equity"]
        yr = e.resample("YE").last().pct_change()
        res[name] = dict(full=metrics(e, out["trades"]),
                         oos=metrics(e[e.index >= "2024-01-01"], out["trades"][out["trades"].entry_time >= "2024-01-01"]),
                         yearly={int(k.year): float(v) for k, v in yr.dropna().items()})
    d = runlog.new_run("meta_xgb")
    b, f = res["base"]["oos"], res["xgb_filtered"]["oos"]
    headline = (f"XGB meta filter OOS(2024+) CAGR {f['cagr']:.1%} DD {f['max_dd']:.1%} Sharpe {f['sharpe']:.2f} "
                f"vs base {b['cagr']:.1%} DD {b['max_dd']:.1%} Sharpe {b['sharpe']:.2f}")
    runlog.save(d, {"test_years": TEST_YEARS, "features": fcols, "label": "r>0", "threshold": "median train proba",
                    "kw": KW}, dict(headline=headline, **res), {"per_year_trades": rep})
    print(rep.round(3).to_string(index=False))
    for k in res:
        print(k, {y: round(v, 3) for y, v in res[k]["yearly"].items()})
    print(headline, "\nsaved", d)


if __name__ == "__main__":
    main()
