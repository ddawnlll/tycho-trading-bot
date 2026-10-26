"""Replay the LIVE decision path (MarketState + experts) bar-by-bar on history and require it to
match the shared-account backtest engine trade-for-trade."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.engine_multi import run_multi
from tycho.ensemble import build_columns, configured_experts, window
from tycho.live import PaperBroker, Position, decide, size
from tycho.marketstate import build_state
from tycho.portfolio import MAJORS, load_universe

WINDOWS = [("2024-02-01", "2024-04-01"), ("2021-11-01", "2022-01-15")]  # 2nd spans the BTC regime flip
DATA_OK = all((Path(__file__).resolve().parent.parent / "data" / f"{s}_1h.parquet").exists() for s in C.UNIVERSE)


def replay(data, start, end, risk):
    bidx = data["BTCUSDT"].index
    idx = bidx[(bidx >= start) & (bidx <= end)]  # same bounds as tycho.ensemble.window
    broker = PaperBroker({"cash": 10_000.0})
    pos: dict[str, Position] = {}
    trades, eq_curve = [], []
    for t in idx:
        prev = {s: df[df.index < t].iloc[-(C.REGIME_WARMUP_BARS if s == "BTCUSDT" else C.WARMUP_BARS):]
                for s, df in data.items() if (df.index < t).sum() > 60}
        dec = decide(prev, pos)
        for s, st in dec.stop_moves.items():
            pos[s].stop = st
        opens = {s: data[s].at[t, "open"] for s in data if t in data[s].index}
        for s in dec.exits:
            p = pos.pop(s)
            broker.sell(p, opens[s])
            trades.append((f"{p.expert}:{s}", p.entry_time, str(t), "signal"))
        for expert, s, stop in dec.entries:
            # the backtest window has no decision bar before its first bar, so it can't enter there
            if t == idx[0] or s in pos or len(pos) >= C.MAX_POSITIONS or s not in opens:
                continue
            eq = broker.equity(pos, opens)
            gross = sum(p.qty * opens[x] for x, p in pos.items())
            est = opens[s] * (1 + (0.0002 if s in MAJORS else 0.0005))
            q = size(eq, gross, est, stop, risk)
            if q <= 0:
                continue
            px, q = broker.buy(s, q, opens[s])
            pos[s] = Position(s, q, px, stop, q * (px - stop), str(t), expert=expert)
        for s in list(pos):
            if t in data[s].index and data[s].at[t, "low"] <= pos[s].stop:
                p = pos.pop(s)
                broker.sell(p, min(data[s].at[t, "open"], p.stop))
                trades.append((f"{p.expert}:{s}", p.entry_time, str(t), "stop"))
        eq_curve.append(broker.equity(pos, {s: data[s].at[t, "close"] for s in pos}))
    return trades, pd.Series(eq_curve, index=idx)


@pytest.mark.skipif(not DATA_OK, reason="historical data not downloaded")
@pytest.mark.parametrize("START,END", WINDOWS)
def test_live_decisions_match_backtest(START, END):
    data = {s: df.assign(funding=0.0) for s, df in load_universe(C.UNIVERSE, C.TIMEFRAME).items()}
    risk = C.RISK_DEFAULT
    cols, sigs, costs, groups = build_columns(data, configured_experts(), build_state(data, C.REGIME_N))
    d, s = window(cols, sigs, START, END)
    bt = run_multi(d, s, costs=costs, groups=groups, max_positions=C.MAX_POSITIONS, max_gross=C.MAX_GROSS,
                   max_lev_pos=C.MAX_LEV_POS, risk=risk)
    bt_tr = bt["trades"][bt["trades"].reason != "eod"]
    bt_set = {(r.symbol, str(r.entry_time), str(r.exit_time), r.reason) for r in bt_tr.itertuples()}
    lv_tr, lv_eq = replay(data, START, END, risk)
    assert len(bt_set) > 5
    assert set(lv_tr) == bt_set
    bt_eq = bt["equity"].reindex(lv_eq.index)
    assert not bt_eq.isna().any() and len(bt_eq) == len(bt["equity"])
    assert float(np.max(np.abs(lv_eq.to_numpy()[:-1] / bt_eq.to_numpy()[:-1] - 1))) < 1e-3
