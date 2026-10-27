"""jesse_momentum_v2 runner. Every rule comes from /repo/preregistration.json.

  python mv2_run.py dev        Jesse backtests on the perp dev chunks (baselines, all grid cells, variants)
  python mv2_run.py analyze    walk-forward selection, criteria, nulls, robustness  -> /results/dev_analysis.json
  python mv2_run.py ml         logistic-regression filter vs. rule baseline           -> /results/ml.json
  python mv2_run.py holdout    once, only for walk-forward survivors                 -> /results/holdout.json
Each Jesse run is cached in /results/cache, so an interrupted stage resumes where it stopped.
"""
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/research_scripts")
import jesse.helpers as jh  # noqa: E402
from jesse.research import backtest  # noqa: E402
from mv2_strategies import BAR_MS, STRATEGIES, features  # noqa: E402

DATA, RES, REPO = Path("/data"), Path("/results"), Path("/repo")
PRE = json.loads((REPO / "preregistration.json").read_text())
UNI = json.loads((DATA / "universe.json").read_text()) if (DATA / "universe.json").exists() else {}
CUTOFF_MS = 1_790_812_800_000  # 2026-10-01 00:00 UTC: last archived month is 2026-09 (Tycho frozen window untouched)
for _c in UNI.get("perp", []):
    _c["end_ms"] = min(_c["end_ms"], CUTOFF_MS)
LEDGER = Path("/ledger/trials.jsonl")
EX, TAKER, WARM, SLOTS, START = "Binance Perpetual Futures", 0.0005, 330, 15, 10_000.0
DAY, MIN = 86_400_000, 60_000
HYPS = {"H1": PRE["hypotheses"]["H1_momentum_pullback"]["grid"],
        "H2": PRE["hypotheses"]["H2_breakout_continuation"]["grid"],
        "H3": PRE["hypotheses"]["H3_relative_strength"]["grid"]}
OOS_FROM = "2021-H1"
RUNS = {"n": 0}


def cells(hyp):
    g = HYPS[hyp]
    a, b = list(g)
    return [{a: x, b: y} for x in g[a] for y in g[b]]


def ckey(params):
    return "_".join(f"{k}{v}" for k, v in params.items()) or "none"


# ---------------------------------------------------------------- ledger
def ledger(row):
    lines = LEDGER.read_text().strip().split("\n")
    row = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "prev": json.loads(lines[-1])["hash"], "git": git_head(), **row}
    row["hash"] = hashlib.sha256(json.dumps(row, sort_keys=True, default=float).encode()).hexdigest()
    with LEDGER.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, default=float) + "\n")


# ---------------------------------------------------------------- data
def load_1m(market, sym, t0, t1):
    """1m Jesse rows in [t0, t1) with internal gaps filled by flat zero-volume candles."""
    parts = []
    d = dt.datetime.utcfromtimestamp(t0 / 1000).replace(day=1)
    while d < dt.datetime.utcfromtimestamp(t1 / 1000):
        p = DATA / "1m" / market / sym / f"{d:%Y-%m}.npy"
        if p.exists():
            parts.append(np.load(p))
        d = (d + dt.timedelta(days=32)).replace(day=1)
    if not parts:
        return np.zeros((0, 6)), 0
    a = np.concatenate(parts)
    a = a[(a[:, 0] >= t0) & (a[:, 0] < t1)]
    if not len(a):
        return a, 0
    a[:, 3] = np.maximum(a[:, 3], a[:, 1:3].max(1))  # repair rare OHLC anomalies (counted in manifest)
    a[:, 4] = np.minimum(a[:, 4], a[:, 1:3].min(1))
    n = int((a[-1, 0] - a[0, 0]) // MIN) + 1
    if n == len(a):
        return a, 0
    full = np.empty((n, 6))
    full[:, 0] = a[0, 0] + MIN * np.arange(n)
    idx = ((a[:, 0] - a[0, 0]) // MIN).astype(int)
    have = np.zeros(n, bool)
    have[idx] = True
    last = np.maximum.accumulate(np.where(have, np.arange(n), 0))
    prev_close = a[np.searchsorted(idx, last), 2]
    full[:, 1:5] = prev_close[:, None]
    full[:, 5] = 0.0
    full[idx] = a
    return full, int(n - len(a))


def bars_4h(m1):
    """Completed 4h Jesse rows from contiguous 1m rows (open-time aligned)."""
    start = int(np.ceil(m1[0, 0] / BAR_MS) * BAR_MS)
    m = m1[m1[:, 0] >= start]
    n = len(m) // 240
    m = m[: n * 240].reshape(n, 240, 6)
    return np.c_[m[:, 0, 0], m[:, 0, 1], m[:, -1, 2], m[:, :, 3].max(1), m[:, :, 4].min(1), m[:, :, 5].sum(1)]


def settlements(market, sym, t0, t1):
    if market == "spot":  # assumed baseline funding 0.01%/8h (preregistration)
        ts = np.arange(int(np.ceil(t0 / (8 * 3_600_000))) * 8 * 3_600_000, t1, 8 * 3_600_000)
        return ts.astype(float), np.full(len(ts), 0.0001)
    rows = []
    d = dt.datetime.utcfromtimestamp(t0 / 1000).replace(day=1)
    while d < dt.datetime.utcfromtimestamp(t1 / 1000):
        p = DATA / "funding" / sym / f"{d:%Y-%m}.npy"
        if p.exists():
            rows.append(np.load(p))
        d = (d + dt.timedelta(days=32)).replace(day=1)
    a = np.concatenate(rows) if rows else np.zeros((0, 2))
    a[:, 0] = a[:, 0] // MIN * MIN
    a = a[(a[:, 0] >= t0) & (a[:, 0] < t1)]
    return a[:, 0], a[:, 1]


def chunk_data(market, ch, mem):
    """Warm-up/trading 1m arrays and stop time for one member of one chunk (None if unusable)."""
    t0, t1 = ch["start_ms"], ch["end_ms"]
    m1, filled = load_1m(market, mem["symbol"], t0 - WARM * BAR_MS, t1)
    if not len(m1) or m1[0, 0] > t0 - WARM * BAR_MS or m1[-1, 0] < t0 + 2 * BAR_MS:
        return None
    warm, trade = m1[m1[:, 0] < t0], m1[m1[:, 0] >= t0]
    stop_at = (trade[-1, 0] + MIN) // BAR_MS * BAR_MS
    return {"warm": warm, "trade": trade, "stop_at": stop_at, "filled": filled, "m1_t0": m1[0, 0], "m1": m1}


# ---------------------------------------------------------------- one Jesse run
def run_one(hyp, params, variant, market, ch, mem, d, schedule=None, model=None):
    sym = mem["symbol"]
    out = RES / "cache" / market / ch["chunk"] / sym / f"{hyp}_{ckey(params)}_{variant}.json"
    if out.exists():
        return json.loads(out.read_text())
    S = STRATEGIES[hyp]
    S.MODEL = model
    if schedule is not None:
        S.SCHEDULE = schedule
    cost = (TAKER + mem["slippage"]) * (3 if variant.endswith("stress") else 1)
    cfg = {"starting_balance": START, "fee": cost, "type": "futures", "futures_leverage": 3,
           "futures_leverage_mode": "cross", "exchange": EX, "warm_up_candles": WARM}
    js = sym[:-4] + "-USDT"
    key = jh.key(EX, js)
    hp = {"stop_at": float(d["stop_at"]), "delay": int(variant.endswith("delay")), **params}
    t = time.time()
    r = backtest(cfg, [{"exchange": EX, "strategy": S, "symbol": js, "timeframe": "4h"}], [],
                 {key: {"exchange": EX, "symbol": js, "candles": d["trade"]}},
                 {key: {"exchange": EX, "symbol": js, "candles": d["warm"]}},
                 hyperparameters=hp, fast_mode=True, generate_equity_curve=True)
    S.MODEL = None
    RUNS["n"] += 1
    st, rate = settlements(market, sym, ch["start_ms"], ch["end_ms"])
    m1, t0 = d["m1"], d["m1_t0"]
    trades, fund_events = [], []
    for tr in r.get("trades", []):
        side = 1 if tr["type"] == "long" else -1
        sel = (st >= tr["opened_at"]) & (st < tr["closed_at"])
        idx = ((st[sel] - t0) // MIN).astype(int).clip(0, len(m1) - 1)
        pay = side * abs(tr["qty"]) * m1[idx, 1] * rate[sel]
        fund_events += list(zip(st[sel].tolist(), pay.tolist()))
        trades.append({"opened_at": tr["opened_at"], "closed_at": tr["closed_at"], "side": side,
                       "qty": abs(tr["qty"]), "entry": tr["entry_price"], "exit": tr["exit_price"],
                       "size": tr["size"], "fee": tr["fee"], "pnl": tr["PNL"], "funding": float(pay.sum())})
    eq = [[p["time"] * 1000, p["value"]] for p in r["equity_curve"][0]["data"]] if r.get("equity_curve") else []
    fin = r["metrics"].get("finishing_balance", START)
    eq.append([float(ch["end_ms"]), fin])
    fe = np.array(sorted(fund_events)).reshape(-1, 2)
    for p in eq:
        p[1] -= fe[fe[:, 0] <= p[0], 1].sum() if len(fe) else 0.0
    res = {"hyp": hyp, "params": params, "variant": variant, "market": market, "chunk": ch["chunk"], "symbol": sym,
           "slippage": mem["slippage"], "filled_minutes": d["filled"], "seconds": round(time.time() - t, 2),
           "open_trades_at_end": r["metrics"].get("total_open_trades", 0), "trades": trades, "equity": eq,
           "funding_total": float(fe[:, 1].sum()) if len(fe) else 0.0, "finishing_balance": fin}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def h3_schedules(market, ch, bars, L, k):
    """Point-in-time weekly ranks among chunk members -> {sym: (decision_ts, side)}."""
    t0, t1 = ch["start_ms"], ch["end_ms"]
    mondays = [t for t in range(t0 - t0 % DAY, t1, DAY) if t >= t0 and dt.datetime.utcfromtimestamp(t / 1000).weekday() == 0]
    sides = {s: [] for s in bars}
    for T in mondays:
        scores = {}
        for s, b in bars.items():
            close_ts = b[:, 0] + BAR_MS
            i = np.searchsorted(close_ts, T)
            if i < len(b) and close_ts[i] == T and i - 6 - L >= 0:
                scores[s] = b[i - 6, 2] / b[i - 6 - L, 2] - 1
        ranked = sorted(scores, key=scores.get, reverse=True)
        for s in bars:
            side = 0
            if len(ranked) >= 2 * k + 1 and s in scores:
                side = 1 if s in ranked[:k] else -1 if s in ranked[-k:] else 0
            sides[s].append(side)
    return {s: (np.array(mondays, float), np.array(v, float)) for s, v in sides.items()}, mondays


def run_market(market, chunk_names, configs, model=None):
    """Run every (hyp, params, variant) config on every member of the given chunks."""
    for ch in [c for c in UNI[market] if c["chunk"] in chunk_names]:
        bars = {}
        for mem in ch["members"]:
            d = chunk_data(market, ch, mem)
            if d is None:
                print("skip unusable", market, ch["chunk"], mem["symbol"], flush=True)
                continue
            b = bars_4h(d["m1"])
            bars[mem["symbol"]] = b
            p = RES / "bars" / market / ch["chunk"] / f"{mem['symbol']}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, b)
            del d
        sched = {}
        for hyp, params, _ in configs:
            if hyp == "H3" and ckey(params) not in sched:
                sched[ckey(params)] = h3_schedules(market, ch, bars, params["L"], params["k"])[0]
        for mem in ch["members"]:
            if mem["symbol"] not in bars:
                continue
            d = chunk_data(market, ch, mem)
            for hyp, params, variant in configs:
                s = sched[ckey(params)][mem["symbol"]] if hyp == "H3" else None
                run_one(hyp, params, variant, market, ch, mem, d, schedule=s, model=model)
            print(f"{market} {ch['chunk']} {mem['symbol']} done (runs this process: {RUNS['n']})", flush=True)
            if RUNS["n"] > PRE["compute_budget"]["jesse_backtests_max"]:
                raise SystemExit("compute budget exhausted")


def dev_configs():
    cfg = [("BH", {}, "base")]
    for hyp in HYPS:
        for p in cells(hyp):
            cfg += [(hyp, p, v) for v in ("base", "stress", "delay")]
    return cfg


# ---------------------------------------------------------------- portfolio metrics
def load_run(market, chunk, sym, hyp, params, variant):
    p = RES / "cache" / market / chunk / sym / f"{hyp}_{ckey(params)}_{variant}.json"
    return json.loads(p.read_text()) if p.exists() else None


def chunk_returns(market, ch, hyp, params, variant, only=None, drop=()):
    """Daily net returns of the 15-slot portfolio over one chunk, plus its runs."""
    n = int((ch["end_ms"] - ch["start_ms"]) // DAY)
    grid = ch["start_ms"] + DAY * (np.arange(n) + 1)
    total = np.full(n, START * SLOTS)
    runs = []
    for mem in ch["members"]:
        if (only and mem["symbol"] not in only) or mem["symbol"] in drop:
            continue
        r = load_run(market, ch["chunk"], mem["symbol"], hyp, params, variant)
        if r is None:
            continue
        runs.append(r)
        e = np.array(r["equity"])
        i = np.searchsorted(e[:, 0], grid, side="right") - 1
        total += np.where(i >= 0, e[i.clip(0), 1], START) - START
    slots = len(only) if only else SLOTS
    total -= START * (SLOTS - slots)
    ret = total / np.r_[START * slots, total[:-1]] - 1
    return grid, ret, runs


def metrics(ret, runs, grid=None):
    ret = np.asarray(ret, float)
    eq = np.cumprod(1 + ret)
    years = len(ret) / 365
    trades = [t for r in runs for t in r["trades"]]
    out = {"days": len(ret), "net_return": float(eq[-1] - 1) if len(eq) else 0.0,
           "cagr": float(eq[-1] ** (1 / years) - 1) if len(eq) and eq[-1] > 0 else -1.0,
           "sharpe": float(ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else 0.0,
           "max_dd": float((1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max()) if len(eq) else 0.0,
           "trades": len(trades),
           "win_rate": float(np.mean([t["pnl"] - t["funding"] > 0 for t in trades])) if trades else None,
           "fees_usd": float(sum(t["fee"] for t in trades)),
           "funding_usd": float(sum(t["funding"] for t in trades)),
           "filled_minutes": int(sum(r["filled_minutes"] for r in runs))}
    if grid is not None and len(ret):
        yrs = np.array([dt.datetime.utcfromtimestamp((g - 1) / 1000).year for g in grid])
        out["by_year"] = {int(y): float(np.prod(1 + ret[yrs == y]) - 1) for y in np.unique(yrs)}
    return out


def stitched(market, chunk_names, pick, variant="base", drop=()):
    """Concatenate chunk portfolios; pick(chunk) -> (hyp, params)."""
    G, R, RUNS_ = [], [], []
    for ch in UNI[market]:
        if ch["chunk"] in chunk_names:
            hyp, params = pick(ch["chunk"])
            g, r, runs = chunk_returns(market, ch, hyp, params, variant, drop=drop)
            G.append(g), R.append(r), RUNS_.extend(runs)
    return np.concatenate(G), np.concatenate(R), RUNS_


def block_bootstrap_q05(ret, block=30, reps=2000, seed=1729):
    rng = np.random.default_rng(seed)
    n = len(ret)
    starts = rng.integers(0, n, (reps, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n] % n
    return float(np.quantile(ret[idx].mean(1), 0.05))


# ---------------------------------------------------------------- nulls (vectorized on the 4h bars)
def bar_funding(market, sym, b, ch):
    st, rate = settlements(market, sym, ch["start_ms"] - WARM * BAR_MS, ch["end_ms"])
    close_ts = b[:, 0] + BAR_MS
    j = np.searchsorted(close_ts, st, side="right") - 1  # settlement in [close_j, close_j+1)
    f = np.zeros(len(b))
    np.add.at(f, j[j >= 0], rate[j >= 0])
    return close_ts, f


def event_null(market, chunk_names, pick, reps=1000, seed=1729, variant="base"):
    rng = np.random.default_rng(seed)
    obs, null = [], np.zeros(reps)
    n_tr = 0
    for ch in UNI[market]:
        if ch["chunk"] not in chunk_names:
            continue
        hyp, params = pick(ch["chunk"])
        for mem in ch["members"]:
            r = load_run(market, ch["chunk"], mem["symbol"], hyp, params, variant)
            if r is None or not r["trades"]:
                continue
            b = np.load(RES / "bars" / market / ch["chunk"] / f"{mem['symbol']}.npy")
            close_ts, f = bar_funding(market, mem["symbol"], b, ch)
            cf = np.r_[0.0, np.cumsum(f)]
            c = b[:, 2]
            lo = np.searchsorted(close_ts, ch["start_ms"])
            hi = np.searchsorted(close_ts, r["equity"][-1][0], side="right") - 1
            cost = 2 * (TAKER + mem["slippage"])
            holds = []
            for t in r["trades"]:
                i0 = int(np.clip(np.searchsorted(close_ts, t["opened_at"]), lo, hi))
                i1 = int(np.clip(np.searchsorted(close_ts, t["closed_at"]), i0 + 1, hi))
                holds.append(i1 - i0)
                obs.append(c[i1] / c[i0] - 1 - cost - (cf[i1] - cf[i0]))
            holds = np.array(holds)
            n_tr += len(holds)
            free = (hi - lo) - holds.sum()
            for rep in range(reps):
                if free < 0:
                    break
                h = rng.permutation(holds)
                gaps = np.diff(np.r_[0, np.sort(rng.integers(0, free + 1, len(h)))])
                s0 = lo + np.cumsum(gaps) + np.r_[0, np.cumsum(h)[:-1]]
                null[rep] += (c[s0 + h] / c[s0] - 1 - cost - (cf[s0 + h] - cf[s0])).sum()
    o = float(np.mean(obs)) if obs else 0.0
    null = null / max(n_tr, 1)
    return {"observed_mean_trade_return": o, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "p": float((1 + (null >= o).sum()) / (1 + reps)), "trades": n_tr}


def h3_weeks(market, ch, L, k):
    """Per chunk: weekly coin returns R[w, s], funding F[w, s], scored mask, actual sides, costs."""
    bars = {m["symbol"]: np.load(p) for m in ch["members"]
            if (p := RES / "bars" / market / ch["chunk"] / f"{m['symbol']}.npy").exists()}
    sched, mondays = h3_schedules(market, ch, bars, L, k)
    syms = list(bars)
    ends = mondays[1:] + [ch["end_ms"]]
    W, N = len(mondays), len(syms)
    R, F, scored = np.zeros((W, N)), np.zeros((W, N)), np.zeros((W, N), bool)
    for j, s in enumerate(syms):
        b = bars[s]
        close_ts, f = bar_funding(market, s, b, ch)
        cf = np.r_[0.0, np.cumsum(f)]
        for w, (T, E) in enumerate(zip(mondays, ends)):
            i = np.searchsorted(close_ts, T)
            scored[w, j] = i < len(b) and close_ts[i] == T and i - 6 - L >= 0
            i1 = min(np.searchsorted(close_ts, E), len(b) - 1)
            if i < len(b) and i1 > i:
                R[w, j], F[w, j] = b[i1, 2] / b[i, 2] - 1, cf[i1] - cf[i]
    sides = np.array([sched[s][1] for s in syms]).T.reshape(W, N)
    cost = np.array([TAKER + next(m["slippage"] for m in ch["members"] if m["symbol"] == s) for s in syms])
    return R, F, scored, sides, cost


def h3_book(R, F, sides, cost, k):
    """Weekly returns per unit gross of a book with weights side/(2k), turnover costs and funding."""
    turn = np.abs(np.diff(np.vstack([np.zeros(sides.shape[1]), sides]), axis=0))
    return ((sides * (R - F)).sum(1) - (turn * cost).sum(1)) / (2 * k)


def h3_null(market, chunk_names, pick, reps=1000, seed=1729):
    rng = np.random.default_rng(seed)
    data = [(h3_weeks(market, c, pick(c["chunk"])[1]["L"], pick(c["chunk"])[1]["k"]), pick(c["chunk"])[1]["k"])
            for c in UNI[market] if c["chunk"] in chunk_names]

    def sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(52)) if x.std() > 0 else 0.0
    o = sharpe(np.concatenate([h3_book(R, F, S, C, k) for (R, F, _, S, C), k in data]))
    null = np.zeros(reps)
    for rep in range(reps):
        books = []
        for (R, F, M, S, C), k in data:
            P = S.copy()
            for w in range(len(S)):
                idx = np.flatnonzero(M[w])
                P[w, idx] = S[w, idx][rng.permutation(len(idx))]
            books.append(h3_book(R, F, P, C, k))
        null[rep] = sharpe(np.concatenate(books))
    return {"observed_weekly_sharpe": o, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "p": float((1 + (null >= o).sum()) / (1 + reps)), "weeks": int(sum(len(d[0][0]) for d in data))}


# ---------------------------------------------------------------- stages
def git_head():
    return os.environ.get("GIT_HEAD")  # set by run.sh from the host checkout


def stage_dev():
    names = [c["chunk"] for c in UNI["perp"]]
    t = time.time()
    run_market("perp", names, dev_configs())
    print("dev stage done in", round(time.time() - t), "s, runs", RUNS["n"], flush=True)


def wf_pick(market, hyp, names):
    """Expanding-window choice per test chunk (Sharpe on all earlier chunks, >= 30 trades)."""
    picks = {}
    for i, name in enumerate(names):
        if name < OOS_FROM:
            continue
        train = names[:i]
        best = None
        for p in cells(hyp):
            g, r, runs = stitched(market, train, lambda _c, p=p: (hyp, p))
            m = metrics(r, runs)
            score = (m["trades"] >= 30, m["sharpe"] if m["trades"] >= 30 else m["trades"])
            if best is None or score > best[0]:
                best = (score, p, m)
        picks[name] = {"params": best[1], "train_sharpe": best[2]["sharpe"], "train_trades": best[2]["trades"]}
    return picks


def stage_analyze():
    names = [c["chunk"] for c in UNI["perp"]]
    oos = [n for n in names if n >= OOS_FROM]
    crit = PRE["walk_forward_pass_criteria_all_required"]
    out = {"chunks": names, "oos_chunks": oos, "baselines": {}, "cells": {}, "walk_forward": {}}
    for label, only in (("BTC_buy_hold", {"BTCUSDT"}), ("equal_weight_buy_hold", None)):
        G, R, RU = [], [], []
        for ch in UNI["perp"]:
            if ch["chunk"] in oos:
                g, r, ru = chunk_returns("perp", ch, "BH", {}, "base", only=only)
                G.append(g), R.append(r), RU.extend(ru)
        out["baselines"][label] = metrics(np.concatenate(R), RU, np.concatenate(G))
        ledger({"stage": "baseline", "name": label, "period": f"{oos[0]}..{oos[-1]}", "metrics": out["baselines"][label]})
    for hyp in HYPS:
        for p in cells(hyp):
            row = {}
            for v in ("base", "stress", "delay"):
                g, r, runs = stitched("perp", oos, lambda _c, p=p: (hyp, p), v)
                row[v] = metrics(r, runs, g)
            out["cells"][f"{hyp}:{ckey(p)}"] = row
            ledger({"stage": "grid_cell_fixed_params", "hyp": hyp, "params": p, "period": f"{oos[0]}..{oos[-1]}",
                    "metrics": {v: {k: m[k] for k in ("cagr", "sharpe", "max_dd", "trades", "net_return")} for v, m in row.items()}})
        picks = wf_pick("perp", hyp, names)
        pick = lambda c, picks=picks, hyp=hyp: (hyp, picks[c]["params"])  # noqa: E731
        g, r, runs = stitched("perp", oos, pick)
        base = metrics(r, runs, g)
        stress = metrics(*stitched("perp", oos, pick, "stress")[1:], None)
        delay = metrics(*stitched("perp", oos, pick, "delay")[1:], None)
        prof = {}
        for run in runs:
            prof[run["symbol"]] = prof.get(run["symbol"], 0.0) + sum(t["pnl"] - t["funding"] for t in run["trades"])
        best_sym = max(prof, key=prof.get) if prof else None
        dropped = metrics(*stitched("perp", oos, pick, drop=(best_sym,))[1:], None)
        q05 = block_bootstrap_q05(r)
        null = h3_null("perp", oos, pick) if hyp == "H3" else event_null("perp", oos, pick)
        pos_cells = sum(out["cells"][f"{hyp}:{ckey(p)}"]["base"]["sharpe"] > 0 for p in cells(hyp))
        yrs = base["by_year"]
        checks = {
            "net_cagr_gt_0": base["cagr"] > crit["net_cagr_gt"],
            "sharpe_ge_0.5": base["sharpe"] >= crit["sharpe_ge"],
            "max_dd_le_0.30": base["max_dd"] <= crit["max_dd_le"],
            "trades_ge_100": base["trades"] >= crit["closed_trades_ge"],
            "stress_3x_cagr_gt_0": stress["cagr"] > crit["stress_3x_cagr_gt"],
            "bootstrap_q05_gt_0": q05 > 0,
            "null_p_le_0.0167": null["p"] <= crit["random_entry_null_p_le"],
            "cells_positive_ge_3of4": pos_cells >= 3,
            "delay_cagr_gt_0": delay["cagr"] > 0,
            "drop_best_symbol_cagr_gt_0": dropped["cagr"] > 0,
            "half_years_nonneg": sum(v >= 0 for v in yrs.values()) >= len(yrs) / 2,
        }
        verdict = "PASS_WALK_FORWARD" if all(checks.values()) else "REJECT"
        out["walk_forward"][hyp] = {"picks": picks, "oos": base, "stress_3x": stress, "delayed_entry": delay,
                                    "drop_best_symbol": {"symbol": best_sym, **dropped}, "bootstrap_q05_daily": q05,
                                    "null": null, "positive_cells": pos_cells, "checks": checks, "verdict": verdict}
        ledger({"stage": "walk_forward", "hyp": hyp, "picks": {k: v["params"] for k, v in picks.items()},
                "oos": {k: base[k] for k in ("cagr", "sharpe", "max_dd", "trades")}, "null_p": null["p"],
                "bootstrap_q05": q05, "checks": checks, "verdict": verdict, "git": git_head()})
        print(hyp, verdict, json.dumps({k: base[k] for k in ("cagr", "sharpe", "max_dd", "trades")}), checks, flush=True)
    (RES / "dev_analysis.json").write_text(json.dumps(out, indent=1, default=float))


def trade_rows(market, chunk, hyp, params, variant="base"):
    """(features, label, opened_at, closed_at, net return) per trade, from the 4h bars Jesse saw."""
    ch = next(c for c in UNI[market] if c["chunk"] == chunk)
    rows = []
    for mem in ch["members"]:
        r = load_run(market, chunk, mem["symbol"], hyp, params, variant)
        if r is None:
            continue
        b = np.load(RES / "bars" / market / chunk / f"{mem['symbol']}.npy")
        close_ts = b[:, 0] + BAR_MS
        for t in r["trades"]:
            i = np.searchsorted(close_ts, t["opened_at"])
            if i < len(b) and close_ts[i] == t["opened_at"] and i >= 330:
                rows.append((features(b[: i + 1]), float(t["pnl"] - t["funding"] > 0), t["opened_at"], t["closed_at"],
                             (t["pnl"] - t["funding"]) / t["size"]))
    return rows


def fold_model(hyp, p, name, names):
    """Purged/embargoed LR fit for one test chunk; (model or None, gate info)."""
    from sklearn.linear_model import LogisticRegression
    ch = next(c for c in UNI["perp"] if c["chunk"] == name)
    train = [row for prev in names if prev < name for row in trade_rows("perp", prev, hyp, p)
             if row[3] <= ch["start_ms"] - 10 * DAY]
    y = np.array([r[1] for r in train])
    days = len({int(r[2] // DAY) for r in train})
    minority = int(min(y.sum(), len(y) - y.sum())) if len(y) else 0
    info = {"train_trades": len(train), "distinct_entry_days": days, "minority": minority,
            "gate": len(train) >= 300 and days >= 150 and minority >= 60}
    if not info["gate"]:
        return None, info
    X = np.array([r[0] for r in train])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    lr = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
    model = {"mean": mu, "std": sd, "coef": lr.coef_[0], "intercept": float(lr.intercept_[0]), "thr": float(y.mean())}
    info["model"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}
    return model, info


def run_with_model(hyp, p, name, model, variants):
    ch = next(c for c in UNI["perp"] if c["chunk"] == name)
    for variant in variants:
        for mem in ch["members"]:
            if (RES / "cache" / "perp" / name / mem["symbol"] / f"{hyp}_{ckey(p)}_{variant}.json").exists():
                continue
            d = chunk_data("perp", ch, mem)
            if d is not None:
                run_one(hyp, p, variant, "perp", ch, mem, d, model=model)


def stage_ml():
    dev = json.loads((RES / "dev_analysis.json").read_text())
    names = dev["chunks"]
    out = {}
    for hyp in ("H1", "H2"):
        picks = dev["walk_forward"][hyp]["picks"]
        folds = {}
        for name in dev["oos_chunks"]:
            p = picks[name]["params"]
            model, folds[name] = fold_model(hyp, p, name, names)
            if model is not None:
                run_with_model(hyp, p, name, model, ("ml", "ml_stress"))
        n_ok = sum(f["gate"] for f in folds.values())
        res = {"folds": folds, "folds_passing_gate": n_ok}
        if n_ok < 6:
            res["verdict"] = "SKIPPED_INSUFFICIENT_OBSERVATIONS"
        else:
            okn = [n for n in dev["oos_chunks"] if folds[n]["gate"]]
            pick = lambda c, picks=picks, hyp=hyp: (hyp, picks[c]["params"])  # noqa: E731
            g, r, runs = stitched("perp", okn, pick, "ml")
            ml = metrics(r, runs, g)
            gb, rb, runsb = stitched("perp", okn, pick, "base")
            rule = metrics(rb, runsb, gb)
            stress = metrics(*stitched("perp", okn, pick, "ml_stress")[1:], None)
            crit = PRE["walk_forward_pass_criteria_all_required"]
            checks = {"net_cagr_gt_0": ml["cagr"] > 0, "sharpe_ge_0.5": ml["sharpe"] >= crit["sharpe_ge"],
                      "max_dd_le_0.30": ml["max_dd"] <= crit["max_dd_le"], "trades_ge_100": ml["trades"] >= 100,
                      "stress_3x_cagr_gt_0": stress["cagr"] > 0, "bootstrap_q05_gt_0": block_bootstrap_q05(r) > 0,
                      "beats_rule_sharpe": ml["sharpe"] > rule["sharpe"]}
            res.update({"chunks": okn, "ml": ml, "rule_same_chunks": rule, "ml_stress_3x": stress, "checks": checks,
                        "verdict": "PASS_ML_COMPARISON" if all(checks.values()) else "REJECT",
                        "note": "random-entry null and bounded robustness were not re-run for the ML variant; "
                                "a PASS here would still need them before any holdout use"})
        out[hyp] = res
        ledger({"stage": "ml_logistic_regression", "hyp": hyp, "folds_passing_gate": n_ok, "verdict": res["verdict"],
                **({"ml": {k: res["ml"][k] for k in ("cagr", "sharpe", "max_dd", "trades")},
                    "rule": {k: res["rule_same_chunks"][k] for k in ("cagr", "sharpe", "max_dd", "trades")}} if "ml" in res else {})})
        print(hyp, res["verdict"], flush=True)
    out["H3"] = {"verdict": "EXCLUDED_BY_PREREGISTRATION"}
    (RES / "ml.json").write_text(json.dumps(out, indent=1, default=float))


def stage_mlcheck():
    """Remaining walk-forward criteria (null, bounded robustness, years) for ML variants that beat their rule."""
    dev = json.loads((RES / "dev_analysis.json").read_text())
    ml = json.loads((RES / "ml.json").read_text())
    names = dev["chunks"]
    crit = PRE["walk_forward_pass_criteria_all_required"]
    for hyp in ("H1", "H2"):
        v = ml[hyp]
        if v["verdict"] != "PASS_ML_COMPARISON":
            continue
        okn, picks = v["chunks"], dev["walk_forward"][hyp]["picks"]
        pick = lambda c, picks=picks, hyp=hyp: (hyp, picks[c]["params"])  # noqa: E731
        for name in okn:
            model, _ = fold_model(hyp, picks[name]["params"], name, names)
            run_with_model(hyp, picks[name]["params"], name, model, ("ml_delay",))
        cell_sharpe = {}
        for p in cells(hyp):  # each fixed cell with its own fold models, on the chunks where its gate passes
            ok = []
            for name in okn:
                model, info = fold_model(hyp, p, name, names)
                if model is not None:
                    run_with_model(hyp, p, name, model, ("ml",))
                    ok.append(name)
            cell_sharpe[ckey(p)] = {"chunks": ok, "sharpe": metrics(*stitched("perp", ok, lambda _c, p=p: (hyp, p), "ml")[1:])["sharpe"] if ok else None}
        g, r, runs = stitched("perp", okn, pick, "ml")
        base = metrics(r, runs, g)
        delay = metrics(*stitched("perp", okn, pick, "ml_delay")[1:], None)
        prof = {}
        for run in runs:
            prof[run["symbol"]] = prof.get(run["symbol"], 0.0) + sum(t["pnl"] - t["funding"] for t in run["trades"])
        best_sym = max(prof, key=prof.get)
        dropped = metrics(*stitched("perp", okn, pick, "ml", drop=(best_sym,))[1:], None)
        null = event_null("perp", okn, pick, variant="ml")
        pos = sum(1 for c in cell_sharpe.values() if c["sharpe"] is not None and c["sharpe"] > 0)
        yrs = base["by_year"]
        checks = {**v["checks"],
                  "null_p_le_0.0167": null["p"] <= crit["random_entry_null_p_le"],
                  "cells_positive_ge_3of4": pos >= 3,
                  "delay_cagr_gt_0": delay["cagr"] > 0,
                  "drop_best_symbol_cagr_gt_0": dropped["cagr"] > 0,
                  "half_years_nonneg": sum(x >= 0 for x in yrs.values()) >= len(yrs) / 2}
        verdict = "PASS_WALK_FORWARD_ML" if all(checks.values()) else "REJECT"
        v["full_walk_forward"] = {"null": null, "cells": cell_sharpe, "positive_cells": pos, "delayed_entry": delay,
                                  "drop_best_symbol": {"symbol": best_sym, **dropped}, "checks": checks, "verdict": verdict}
        v["verdict"] = verdict
        v["note"] = "all preregistered walk-forward criteria evaluated for the ML variant (stage mlcheck)"
        ledger({"stage": "ml_full_walk_forward", "hyp": hyp, "null_p": null["p"], "positive_cells": pos,
                "delay_cagr": delay["cagr"], "drop_best": [best_sym, dropped["cagr"]], "checks": checks, "verdict": verdict})
        print(hyp, verdict, checks, flush=True)
    (RES / "ml.json").write_text(json.dumps(ml, indent=1, default=float))


def stage_holdout():
    marker = RES / "holdout.json"
    if marker.exists():
        raise SystemExit("holdout already read once; refusing to re-run (preregistration)")
    dev = json.loads((RES / "dev_analysis.json").read_text())
    ml = json.loads((RES / "ml.json").read_text())
    survivors = [(h, False) for h, v in dev["walk_forward"].items() if v["verdict"] == "PASS_WALK_FORWARD"]
    survivors += [(h, True) for h in ("H1", "H2") if ml.get(h, {}).get("verdict") == "PASS_WALK_FORWARD_ML"]
    out = {"survivors": [f"{h}{'+LR' if m else ''}" for h, m in survivors], "results": {}}
    if not survivors:
        out["note"] = "No candidate passed walk-forward; the holdout was NOT read and stays untouched."
        marker.write_text(json.dumps(out, indent=1))
        ledger({"stage": "holdout", "survivors": [], "read": False})
        return
    names = [c["chunk"] for c in UNI["spot"]]
    crit = PRE["holdout_pass_criteria_all_required"]
    for hyp, use_ml in survivors:
        label = hyp + ("+LR" if use_ml else "")
        full = max(cells(hyp), key=lambda p: metrics(*stitched("perp", dev["chunks"], lambda _c: (hyp, p))[1:])["sharpe"])
        model, info = None, None
        if use_ml:  # fit on every dev trade of the frozen cell (no embargo needed: the holdout is 2018-19)
            from sklearn.linear_model import LogisticRegression
            rows = [row for n in dev["chunks"] for row in trade_rows("perp", n, hyp, full)]
            X, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
            mu, sd = X.mean(0), X.std(0) + 1e-12
            lr = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
            model = {"mean": mu, "std": sd, "coef": lr.coef_[0], "intercept": float(lr.intercept_[0]), "thr": float(y.mean())}
            info = {"train_trades": len(rows), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}}
        variants = ("ml", "ml_stress") if use_ml else ("base", "stress")
        ledger({"stage": "holdout_params_frozen", "hyp": label, "params": full, "model": info})
        run_market("spot", names, [(hyp, full, v) for v in variants], model=model)
        g, r, runs = stitched("spot", names, lambda _c: (hyp, full), variants[0])
        m = metrics(r, runs, g)
        st = metrics(*stitched("spot", names, lambda _c: (hyp, full), variants[1])[1:], None)
        checks = {"net_return_gt_0": m["net_return"] > crit["net_return_gt"], "sharpe_ge_0.3": m["sharpe"] >= crit["sharpe_ge"],
                  "max_dd_le_0.30": m["max_dd"] <= crit["max_dd_le"], "stress_return_gt_0": st["net_return"] > 0,
                  "trades_ge_30": m["trades"] >= crit["closed_trades_ge"]}
        v = "CANDIDATE_FOR_NEW_PREREGISTERED_PROSPECTIVE_PAPER_TRIAL" if all(checks.values()) else "REJECT"
        out["results"][label] = {"params": full, "model": info, "holdout": m, "stress_3x": st, "checks": checks, "verdict": v}
        ledger({"stage": "holdout", "hyp": label, "params": full,
                "metrics": {k: m[k] for k in ("net_return", "sharpe", "max_dd", "trades")}, "checks": checks, "verdict": v})
        print(label, v, checks, flush=True)
    marker.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"dev": stage_dev, "analyze": stage_analyze, "ml": stage_ml, "mlcheck": stage_mlcheck,
     "holdout": stage_holdout}[sys.argv[1]]()
