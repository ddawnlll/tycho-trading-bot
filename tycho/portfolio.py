"""Equal-capital sub-account portfolio: each symbol trades its own slice of capital independently."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load, resample
from .engine import Costs, metrics, run
from .strategies import STRATEGIES, apply_regime, ema

MAJORS = {"BTCUSDT", "ETHUSDT"}


def load_universe(symbols, tf):
    out = {}
    for s in symbols:
        try:
            out[s] = resample(load(s), tf)
        except FileNotFoundError:
            pass
    return out


def regime(btc: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    e = ema(btc["close"], n)
    return (btc["close"] > e), (btc["close"] < e)


def backtest(data: dict, strategy: str, params: dict, start: str, end: str, risk: float, lev: float,
             regime_n: int | None = None, capital: float = 10_000.0):
    per = capital / len(data)
    eqs, trades, per_sym = {}, [], {}
    bull = bear = None
    if regime_n:
        bull, bear = regime(data["BTCUSDT"], regime_n)
    for s, df in data.items():
        sig = STRATEGIES[strategy](df, **params)
        if regime_n:
            b = bull.reindex(df.index).fillna(False).to_numpy(bool)
            r = bear.reindex(df.index).fillna(False).to_numpy(bool)
            sig = apply_regime(sig, b, r)
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() < 50:
            continue
        sl = type(sig)(*(getattr(sig, f)[mask] for f in sig.__dataclass_fields__))
        costs = Costs(slippage=0.0002 if s in MAJORS else 0.0005)
        res = run(df[mask], sl, risk=risk, max_lev=lev, costs=costs, equity0=per)
        eqs[s] = res["equity"]
        if len(res["trades"]):
            trades.append(res["trades"].assign(symbol=s))
        m = metrics(res["equity"], res["trades"])
        per_sym[s] = dict(ret=m["total_return"], dd=m["max_dd"], n=m["n_trades"])
    idx = pd.DatetimeIndex(sorted(set().union(*[e.index for e in eqs.values()])))
    idle = (len(data) - len(eqs)) * per
    total = sum(e.reindex(idx).ffill().fillna(per) for e in eqs.values()) + idle
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return total, tr, metrics(total, tr), per_sym
