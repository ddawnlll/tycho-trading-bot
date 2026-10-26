"""Experts: each trades one idea, only in the market state it requests from MarketState.

An expert = (signals(df, st) -> Signals, requires(st) -> bool mask). Entries are ANDed with the
mask; exits/stops are not (a position is always managed until its stop/exit). `st` is the
MarketState frame for that symbol (MarketState.for_symbol), so experts never compute regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .engine import Signals
from .strategies import _arr, atr, donchian, ema

def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _sig(n, le=None, se=None, ls=None, ss=None, lt=None, st=None, lx=None, sx=None) -> Signals:
    f, nan = np.zeros(n, bool), np.full(n, np.nan)
    b = lambda x: f.copy() if x is None else _arr(x).astype(bool)  # noqa: E731
    v = lambda x: nan.copy() if x is None else _arr(x)  # noqa: E731
    return Signals(b(le), b(se), v(ls), v(ss), v(lt), v(st), b(lx), b(sx))


# ---------------------------------------------------------------------------------- signal rules
def trend_long(df, st, entry_n=30, exit_n=30, stop_atr=3.0):
    return donchian(df, entry_n, exit_n, stop_atr, allow_short=False)


def trend_short(df, st, entry_n=30, exit_n=15, stop_atr=2.5):
    s = donchian(df, entry_n, exit_n, stop_atr, allow_short=True)
    n = len(df)
    return Signals(np.zeros(n, bool), s.short_entry, s.long_stop, s.short_stop, s.long_trail, s.short_trail,
                   s.long_exit, s.short_exit)


def range_revert(df, st, rsi_n=2, rsi_lo=10.0, stop_atr=2.0, exit_n=10):
    """Buy short-term washouts in a ranging market; exit on reversion to the mean."""
    c = df["close"]
    a = atr(df)
    le = (rsi(c, rsi_n) < rsi_lo) & (c < c.rolling(20).mean() - 1.5 * c.rolling(20).std())
    return _sig(len(df), le=le, ls=c - stop_atr * a, lx=c > c.rolling(exit_n).mean())


def squeeze_breakout(df, st, bb_n=20, sq_pct=0.15, lookback=6, trail_atr=3.0, trail_n=20):
    """Volatility contraction then expansion: band width in its bottom percentile recently,
    then a close above the upper band."""
    c = df["close"]
    a = atr(df)
    mid, sd = c.rolling(bb_n).mean(), c.rolling(bb_n).std()
    width = (4 * sd) / mid
    squeezed = (width.rolling(360, min_periods=180).rank(pct=True) < sq_pct).astype(int).rolling(lookback).max() > 0
    le = squeezed & (c > mid + 2 * sd)
    return _sig(len(df), le=le, ls=mid - 1.0 * a, lt=df["high"].rolling(trail_n).max() - trail_atr * a)


def funding_squeeze(df, st, stop_atr=2.5, trail_atr=3.0, trail_n=20):
    """Shorts crowded (negative funding z) while price holds up -> long the squeeze.
    Trigger: close above prior bar high and above EMA50."""
    c = df["close"]
    a = atr(df)
    le = (c > df["high"].shift()) & (c > ema(c, 50))
    return _sig(len(df), le=le, ls=c - stop_atr * a, lt=df["high"].rolling(trail_n).max() - trail_atr * a)


def xs_momentum(df, st, top_k=3, exit_rank=6, stop_atr=4.0):
    """Hold the strongest coins by 30-day return; leave when they drop out of the top group.
    Rank comes from MarketState (cross-sectional), evaluated daily at the 00:00 bar."""
    c = df["close"]
    a = atr(df)
    daily = df.index.hour == 0
    r = st["mom_rank"]
    le = daily & (r <= top_k) & (c > ema(c, 50))
    lx = daily & (r > exit_rank)
    return _sig(len(df), le=le, ls=c - stop_atr * a, lx=lx.fillna(False))


# ---------------------------------------------------------------------------------- state requests
def req_bull(st):
    return st["btc_bull"] & st["listed"]


def req_bear_weak(st, max_breadth=0.4):
    return ~st["btc_bull"] & (st["breadth"] < max_breadth) & st["listed"]


def req_range(st):
    return (st["trend"] == 0) & st["btc_bull"] & st["listed"]


def req_low_vol_bull(st, max_vol_pct=0.35):
    return (st["vol_pct"] < max_vol_pct) & st["btc_bull"] & st["listed"]


def req_shorts_crowded(st, z=-1.5):
    return (st["funding_z"] < z) & st["btc_bull"] & st["listed"]


def req_strong_breadth(st, min_breadth=0.5):
    return st["btc_bull"] & (st["breadth"] > min_breadth) & st["listed"]


@dataclass
class Expert:
    name: str
    rule: Callable
    requires: Callable
    params: dict = field(default_factory=dict)
    req_params: dict = field(default_factory=dict)

    def signals(self, df: pd.DataFrame, st: pd.DataFrame) -> Signals:
        # Isolate plugin inputs: an expert cannot corrupt another expert's snapshot.
        s = self.rule(df.copy(deep=True), st.copy(deep=True), **self.params)
        gate = self.requires(st.copy(deep=True), **self.req_params).fillna(False).to_numpy(bool)
        if any(len(getattr(s, f)) != len(df) for f in s.__dataclass_fields__):
            raise ValueError(f"{self.name}: signal shape does not match snapshot")
        return Signals(s.long_entry & gate, s.short_entry & gate, *(getattr(s, f) for f in
                       ("long_stop", "short_stop", "long_trail", "short_trail", "long_exit", "short_exit")))


CATALOG = {
    "trend_long": Expert("trend_long", trend_long, req_bull),
    "trend_short": Expert("trend_short", trend_short, req_bear_weak),
    "range_revert": Expert("range_revert", range_revert, req_range),
    "squeeze_breakout": Expert("squeeze_breakout", squeeze_breakout, req_low_vol_bull),
    "funding_squeeze": Expert("funding_squeeze", funding_squeeze, req_shorts_crowded),
    "xs_momentum": Expert("xs_momentum", xs_momentum, req_strong_breadth),
}
