"""Momentum/trend expert candidates (pre-registered 2026-10-27) on the existing Expert/MarketState
architecture. New file on purpose: the frozen files in research/protocol.json stay byte-identical.

Each candidate is a hypothesis taken from a championship trader's described method, not alpha:
  leader_breakout  Luk / Tito / Minervini: the strongest coins (MarketState momentum rank) in a strong
                   market (BTC bull + breadth) breaking out of a tight base; tight stop, wide trail.
  tsmom            Trend CTAs / Larry Williams mechanical trend: 30/90/180-day momentum vote, daily.
  dual_momentum    J Law regime switching + Antonacci: weekly, relative leaders that also have
                   positive absolute 90-day momentum.
  vol_expansion    Larry Williams volatility breakout: a wide-range expansion bar closing near its
                   high in a bull regime; short trail.
All rules use bars <= i only (tests/test_momentum.py truncation test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Costs, Signals
from .experts import CATALOG, Expert, _sig
from .marketstate import MarketState, build_state
from .strategies import atr, ema

DAY = 6                       # 4h bars per day
MIN_QVOL_USD = 20e6           # point-in-time tradability: 30-day median daily quote volume


# ---------------------------------------------------------------------------------- rules
def leader_breakout(df, st, n=20, base_max=0.35, stop_atr=1.5, trail_atr=3.0, trail_n=20):
    c, h, l = df["close"], df["high"], df["low"]
    a = atr(df)
    hi, lo = h.shift().rolling(n).max(), l.shift().rolling(n).min()
    tight = (hi - lo) / c < base_max
    le = (c > hi) & tight
    return _sig(len(df), le=le, ls=c - stop_atr * a, lt=h.rolling(trail_n).max() - trail_atr * a)


def tsmom(df, st, stop_atr=4.0, horizons=(30, 90, 180)):
    c = df["close"]
    a = atr(df)
    votes = sum(np.sign(np.log(c / c.shift(d * DAY))) for d in horizons)
    daily = pd.Series(df.index.hour == 0, index=df.index)
    le = daily & (votes == len(horizons))
    lx = daily & (votes <= 0)
    return _sig(len(df), le=le, ls=c - stop_atr * a, lx=lx.fillna(False))


def dual_momentum(df, st, top_k=3, exit_rank=6, stop_atr=5.0):
    c = df["close"]
    a = atr(df)
    weekly = pd.Series((df.index.dayofweek == 0) & (df.index.hour == 0), index=df.index)
    absmom = np.log(c / c.shift(90 * DAY))
    r = st["mom_rank"]
    le = weekly & (r <= top_k) & (absmom > 0)
    lx = weekly & ((r > exit_rank) | (absmom < 0))
    return _sig(len(df), le=le, ls=c - stop_atr * a, lx=lx.fillna(False))


def vol_expansion(df, st, k=1.5, close_pos=0.8, trail_atr=2.0, trail_n=6):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = atr(df).shift()  # ATR before the expansion bar
    rng = (h - l).replace(0, np.nan)
    le = ((c - o) > k * a) & ((c - l) / rng >= close_pos) & (c > ema(c, 50))
    return _sig(len(df), le=le, ls=l, lt=h.rolling(trail_n).max() - trail_atr * atr(df))


# ---------------------------------------------------------------------------------- requested states
def req_leader(st, top_k=4, min_breadth=0.3):
    return st["btc_bull"] & (st["breadth"] > min_breadth) & (st["mom_rank"] <= top_k) & st["listed"] & st["liquid"]


def req_bull_liquid(st):
    return st["btc_bull"] & st["listed"] & st["liquid"]


def req_any_liquid(st):
    return st["listed"] & st["liquid"]


RULES = {"leader_breakout": (leader_breakout, req_leader), "tsmom": (tsmom, req_bull_liquid),
         "dual_momentum": (dual_momentum, req_bull_liquid), "vol_expansion": (vol_expansion, req_bull_liquid),
         "trend_long": (CATALOG["trend_long"].rule, req_bull_liquid)}

# Pre-registered grids (rule params, requires params). Walk-forward picks one cell per year on
# PAST data only. No cell is added after seeing any result.
GRIDS = {
    "leader_breakout": [({"n": n}, {"top_k": k}) for n in (20, 40) for k in (3, 5)],
    "tsmom": [({"stop_atr": s}, {}) for s in (3.0, 5.0)] + [({"stop_atr": s}, {"_gate": "any"}) for s in (3.0, 5.0)],
    "dual_momentum": [({"top_k": k, "exit_rank": e}, {}) for k in (2, 3) for e in (5, 8)],
    "vol_expansion": [({"k": k, "trail_atr": t}, {}) for k in (1.5, 2.0) for t in (2.0, 3.0)],
    "trend_long": [({}, {})],  # incumbent baseline, fixed
}


def make_expert(name: str, params: dict, req: dict) -> Expert:
    rule, requires = RULES[name]
    req = dict(req)
    if req.pop("_gate", None) == "any":
        requires = req_any_liquid
    return Expert(name, rule, requires, dict(params), req)


# ---------------------------------------------------------------------------------- data -> columns
def liquid_mask(df: pd.DataFrame) -> pd.Series:
    """30-day median daily quote volume >= MIN_QVOL_USD, known at bar close (causal)."""
    daily_qvol = df["qvol"].rolling(DAY).sum()
    return (daily_qvol.rolling(30 * DAY, min_periods=30 * DAY).median() >= MIN_QVOL_USD).fillna(False)


def build(data: dict[str, pd.DataFrame], experts: list[Expert], ms: MarketState | None = None,
          tradable: list[str] | None = None, slip_major=0.0002, slip_alt=0.0005, majors=("BTCUSDT", "ETHUSDT")):
    """Like tycho.ensemble.build_columns, plus the point-in-time liquidity gate and a tradable list
    (holdout: BTC drives the regime but is not traded)."""
    ms = ms or build_state(data)
    tradable = tradable or list(data)
    cols, sigs, costs, groups = {}, {}, {}, {}
    for ex in experts:
        for s in tradable:
            df = data[s]
            st = ms.for_symbol(s).assign(liquid=liquid_mask(df))
            key = f"{ex.name}:{s}"
            cols[key], sigs[key] = df, ex.signals(df, st)
            costs[key] = Costs(slippage=slip_major if s in majors else slip_alt)
            groups[key] = s
    return cols, sigs, costs, groups


def window(cols, sigs, start, end):
    d, s = {}, {}
    for k, df in cols.items():
        m = (df.index >= start) & (df.index <= end)
        if m.sum() > 50:
            d[k] = df[m]
            s[k] = Signals(*(getattr(sigs[k], f)[m] for f in Signals.__dataclass_fields__))
    return d, s
