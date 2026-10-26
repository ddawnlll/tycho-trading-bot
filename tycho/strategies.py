"""Signal generators. Every array at index i uses only bars <= i (causal rolling/ewm)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Signals


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _arr(s) -> np.ndarray:
    return np.asarray(s, dtype=float)


def trend_pullback(df: pd.DataFrame, fast: int = 9, mid: int = 21, slow: int = 50,
                   hi_lookback: int = 20, max_pb_bars: int = 8, stop_atr: float = 0.5,
                   trail_atr: float = 3.0, trail_n: int = 10, exit_on_mid: bool = False,
                   allow_short: bool = True) -> Signals:
    """Martin-Luk-style: strong trend, buy the first controlled pullback to the fast/mid EMA zone,
    enter when price reclaims the prior bar high, stop under the pullback low, trail with a chandelier.

    Long setup at close i:
      trend:   ema_mid > ema_slow, close > ema_slow, ema_slow rising over 5 bars
      impulse: a fresh hi_lookback-bar high was printed within the last max_pb_bars bars
      pullback: since that high, a low touched ema_mid zone (low <= ema_fast) but no close below ema_slow
      trigger: close > previous high (reclaim) and close > ema_fast
    """
    c, h, l = df["close"], df["high"], df["low"]
    ef, em, es = ema(c, fast), ema(c, mid), ema(c, slow)
    a = atr(df)
    out = {}
    for side in (1, -1):
        if side == 1:
            trend = (em > es) & (c > es) & (es > es.shift(5))
            new_ext = h >= h.rolling(hi_lookback).max()
            touched = l <= ef
            broke = c < es
            trigger = (c > h.shift()) & (c > ef)
            pb_low = l.rolling(max_pb_bars).min()
            stop = pb_low - stop_atr * a
            trail = h.rolling(trail_n).max() - trail_atr * a
            exit_sig = c < em if exit_on_mid else pd.Series(False, index=df.index)
        else:
            trend = (em < es) & (c < es) & (es < es.shift(5))
            new_ext = l <= l.rolling(hi_lookback).min()
            touched = h >= ef
            broke = c > es
            trigger = (c < l.shift()) & (c < ef)
            pb_high = h.rolling(max_pb_bars).max()
            stop = pb_high + stop_atr * a
            trail = l.rolling(trail_n).min() + trail_atr * a
            exit_sig = c > em if exit_on_mid else pd.Series(False, index=df.index)
        # bars since last fresh extreme
        idx = pd.Series(np.where(new_ext, np.arange(len(df)), np.nan), index=df.index).ffill()
        since = np.arange(len(df)) - idx
        recent = (since >= 1) & (since <= max_pb_bars)
        touched_since = touched.astype(int).rolling(max_pb_bars).max().astype(bool)
        broke_since = broke.astype(int).rolling(max_pb_bars).max().astype(bool)
        entry = trend & recent & touched_since & ~broke_since & trigger
        out[side] = (_arr(entry.fillna(False)).astype(bool), _arr(stop), _arr(trail), _arr(exit_sig).astype(bool))
    if not allow_short:
        out[-1] = (np.zeros(len(df), bool),) + out[-1][1:]
    return Signals(out[1][0], out[-1][0], out[1][1], out[-1][1], out[1][2], out[-1][2], out[1][3], out[-1][3])


def donchian(df: pd.DataFrame, entry_n: int = 55, exit_n: int = 20, stop_atr: float = 2.5,
             allow_short: bool = True) -> Signals:
    """Classic turtle-style breakout: enter on close beyond entry_n extreme, initial stop
    stop_atr*ATR, exit on exit_n-bar opposite extreme (as trailing stop)."""
    c, h, l = df["close"], df["high"], df["low"]
    a = atr(df, 20)
    le = c > h.shift().rolling(entry_n).max()
    se = c < l.shift().rolling(entry_n).min()
    if not allow_short:
        se[:] = False
    z = np.zeros(len(df), bool)
    return Signals(_arr(le).astype(bool), _arr(se).astype(bool), _arr(c - stop_atr * a), _arr(c + stop_atr * a),
                   _arr(l.rolling(exit_n).min()), _arr(h.rolling(exit_n).max()), z, z)


def buy_hold(df: pd.DataFrame) -> Signals:
    n = len(df)
    le = np.ones(n, bool)  # (re)enter whenever flat -> works on any slice
    z = np.zeros(n, bool)
    nan = np.full(n, np.nan)
    return Signals(le, z, np.full(n, 1e-9), nan, nan, nan, z, z)


STRATEGIES = {"trend_pullback": trend_pullback, "donchian": donchian, "buy_hold": buy_hold}


def ma_trend(df: pd.DataFrame, fast: int = 50, slow: int = 200, stop_atr: float = 3.0,
             trail_atr: float = 4.0, trail_n: int = 20, allow_short: bool = True) -> Signals:
    """Time-series momentum: in trend while close > slow EMA and fast EMA > slow EMA.
    Enter on the bar the state turns on; exit on close back through the slow EMA or a wide
    chandelier stop. Stop distance (stop_atr*ATR) also sets volatility-scaled size."""
    c, h, l = df["close"], df["high"], df["low"]
    ef, es = ema(c, fast), ema(c, slow)
    a = atr(df)
    up = (c > es) & (ef > es)
    dn = (c < es) & (ef < es)
    le = up & ~up.shift(fill_value=False)
    se = dn & ~dn.shift(fill_value=False)
    if not allow_short:
        se[:] = False
    return Signals(_arr(le).astype(bool), _arr(se).astype(bool), _arr(c - stop_atr * a), _arr(c + stop_atr * a),
                   _arr(h.rolling(trail_n).max() - trail_atr * a), _arr(l.rolling(trail_n).min() + trail_atr * a),
                   _arr(c < es).astype(bool), _arr(c > es).astype(bool))


STRATEGIES["ma_trend"] = ma_trend


def apply_regime(sig: Signals, bull: np.ndarray, bear: np.ndarray) -> Signals:
    """Market-regime gate (J Law idea): longs only in a BTC bull regime, shorts only in a bear one."""
    return Signals(sig.long_entry & bull, sig.short_entry & bear, sig.long_stop, sig.short_stop,
                   sig.long_trail, sig.short_trail, sig.long_exit, sig.short_exit)
