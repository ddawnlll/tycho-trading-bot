"""Replay the LIVE decision path bar-by-bar on history and require it to match the backtest engine."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.engine_multi import run_multi
from tycho.live import PaperBroker, Position, decide, size
from tycho.portfolio import MAJORS, load_universe
from scripts.multi_research import build, window

WINDOWS = [("2024-01-01", "2024-05-01"), ("2021-10-01", "2022-04-01")]  # 2nd spans the BTC regime flip
DATA_OK = all((Path(__file__).resolve().parent.parent / "data" / f"{s}_1h.parquet").exists() for s in C.UNIVERSE)


def replay(data, start, end, risk):
    syms = list(data)
    bidx = data["BTCUSDT"].index
    idx = bidx[(bidx >= start) & (bidx <= end)]  # same bounds as scripts.multi_research.window
    broker = PaperBroker({"cash": 10_000.0})
    pos: dict[str, Position] = {}
    trades = []
    eq_curve = []
    for t in idx:
        # decision at close of the previous bar -> act at open t
        prev = {s: df[df.index < t].iloc[-(C.REGIME_WARMUP_BARS if s == "BTCUSDT" else C.WARMUP_BARS):]
                for s, df in data.items() if (df.index < t).sum() > 60}
        dec = decide(prev, pos)
        for s, st in dec.stop_moves.items():
            pos[s].stop = st
        opens = {s: data[s].at[t, "open"] for s in syms if t in data[s].index}
        for s, stop in dec.entries:
            # the backtest window has no decision bar before its first bar, so it can't enter there
            if t == idx[0] or len(pos) >= C.MAX_POSITIONS or s not in opens:
                continue
            eq = broker.equity(pos, opens)
            gross = sum(p.qty * opens[x] for x, p in pos.items())
            est = opens[s] * (1 + (0.0002 if s in MAJORS else 0.0005))
            q = size(eq, gross, est, stop, risk)
            if q <= 0:
                continue
            px, q = broker.buy(s, q, opens[s])
            pos[s] = Position(s, q, px, stop, q * (px - stop), str(t))
        for s in list(pos):
            if t in data[s].index:
                bar = data[s].loc[t]
                if bar.low <= pos[s].stop:
                    p = pos.pop(s)
                    broker.sell(p, min(bar.open, p.stop))
                    trades.append((s, p.entry_time, str(t)))
        closes = {s: data[s].at[t, "close"] for s in pos}
        eq_curve.append(broker.equity(pos, closes))
    return trades, pd.Series(eq_curve, index=idx)


@pytest.mark.skipif(not DATA_OK, reason="historical data not downloaded")
@pytest.mark.parametrize("START,END", WINDOWS)
def test_live_decisions_match_backtest(START, END):
    data = {s: df.assign(funding=0.0) for s, df in load_universe(C.UNIVERSE, C.TIMEFRAME).items()}
    risk = C.RISK_DEFAULT
    sigs, ranks, costs = build(data, C.STRATEGY, C.PARAMS, C.REGIME_N)
    d, s, _ = window(data, sigs, ranks, START, END)
    bt = run_multi(d, s, costs=costs, max_positions=C.MAX_POSITIONS, max_gross=C.MAX_GROSS,
                   max_lev_pos=C.MAX_LEV_POS, risk=risk)
    bt_tr = bt["trades"][bt["trades"].reason == "stop"]
    bt_set = {(r.symbol, str(r.entry_time), str(r.exit_time)) for r in bt_tr.itertuples()}
    lv_tr, lv_eq = replay(data, START, END, risk)
    assert len(bt_set) > 5
    assert set(lv_tr) == bt_set
    # equity paths agree (both ignore funding here); allow tiny float drift from EWM warmup
    bt_eq = bt["equity"].reindex(lv_eq.index)
    assert not bt_eq.isna().any() and len(bt_eq) == len(bt["equity"])
    assert float(np.max(np.abs(lv_eq.to_numpy()[:-1] / bt_eq.to_numpy()[:-1] - 1))) < 1e-3
