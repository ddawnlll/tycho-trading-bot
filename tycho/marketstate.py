"""MarketState service: causal market-regime labels computed at each 4h bar close.

Experts don't compute regimes themselves; they *request* the state from this service and declare
which states they trade in (see tycho/experts.py). Every value at time t uses bars <= t only.

Global (one value per timestamp, from BTC + cross-section of the universe):
  btc_bull      BTC close > EMA(1200)  (~200 days on 4h)
  breadth       share of symbols with >= 1000 bars of history (converged EMA) with close > own EMA(200)
  btc_vol_pct   percentile of BTC ATR% within its trailing 1-year window (0..1)
Per symbol:
  trend         +1 up / -1 down / 0 range: EMA50 vs EMA200 direction, but 0 when the 30-bar
                Kaufman efficiency ratio < ER_RANGE (choppy)
  er            efficiency ratio (0 = pure noise, 1 = straight line)
  vol_pct       percentile of own ATR% in trailing 1-year window
  funding_z     z-score of 3-day funding sum vs trailing 90 days (crowding: >0 longs pay)
  mom_rank      cross-sectional rank of 30-day return among listed symbols (1 = strongest)

Caveat: the 12-coin universe was chosen in 2026, so breadth/rank carry survivorship bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import atr, ema

YEAR = 6 * 365          # 4h bars per year
ER_RANGE = 0.25
MIN_HIST = 200          # bars before a symbol is tradeable / ranked
BREADTH_HIST = 1000     # bars before its EMA200 has converged enough to count in breadth


def efficiency_ratio(c: pd.Series, n: int = 30) -> pd.Series:
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def trailing_pct(s: pd.Series, n: int = YEAR) -> pd.Series:
    """Percentile of the current value within the trailing window (inclusive). Causal."""
    return s.rolling(n, min_periods=n).rank(pct=True)  # full window only: NaN in the first year


@dataclass
class MarketState:
    index: pd.DatetimeIndex
    btc_bull: pd.Series
    breadth: pd.Series
    btc_vol_pct: pd.Series
    sym: dict[str, pd.DataFrame]   # per symbol: trend, er, vol_pct, funding_z, mom_rank, listed

    def for_symbol(self, symbol: str) -> pd.DataFrame:
        """Global + per-symbol state aligned to that symbol's bars."""
        s = self.sym[symbol]
        return s.assign(btc_bull=self.btc_bull.reindex(s.index).fillna(False).astype(bool),
                        breadth=self.breadth.reindex(s.index),
                        btc_vol_pct=self.btc_vol_pct.reindex(s.index))

    def snapshot(self, t: pd.Timestamp) -> dict:
        """Human-readable state at bar t (what the live bot logs / experts request)."""
        out = dict(time=str(t), btc_bull=bool(self.btc_bull.get(t, False)),
                   breadth=float(self.breadth.get(t, np.nan)), btc_vol_pct=float(self.btc_vol_pct.get(t, np.nan)))
        out["symbols"] = {s: {k: (float(v) if pd.notna(v) else None) for k, v in df.loc[t].items()}
                          for s, df in self.sym.items() if t in df.index}
        return out


def build_state(data: dict[str, pd.DataFrame], regime_n: int = 1200) -> MarketState:
    btc = data["BTCUSDT"]
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    btc_bull = (btc["close"] > ema(btc["close"], regime_n))
    btc_vol_pct = trailing_pct(atr(btc) / btc["close"])
    above, listed, ret30 = {}, {}, {}
    sym = {}
    for s, df in data.items():
        c = df["close"]
        n_hist = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        ok = n_hist >= MIN_HIST
        e50, e200 = ema(c, 50), ema(c, 200)
        er = efficiency_ratio(c)
        direction = np.sign(e50 - e200)
        trend = direction.where(er >= ER_RANGE, 0.0).where(ok)
        fund3 = df["funding"].rolling(18).sum() if "funding" in df else pd.Series(0.0, index=df.index)
        fz = (fund3 - fund3.rolling(540).mean()) / fund3.rolling(540).std()
        sym[s] = pd.DataFrame({"trend": trend, "er": er, "vol_pct": trailing_pct(atr(df) / c),
                               "funding_z": fz, "listed": ok})
        above[s] = (c > e200).where(n_hist >= BREADTH_HIST).reindex(idx)
        listed[s] = ok.reindex(idx).fillna(False)
        ret30[s] = np.log(c / c.shift(180)).where(ok).reindex(idx)
    A = pd.DataFrame(above).astype(float)
    breadth = A.mean(axis=1, skipna=True)  # only symbols with history at t
    R = pd.DataFrame(ret30)
    ranks = R.rank(axis=1, ascending=False)  # NaN (unlisted) excluded per row
    for s in sym:
        sym[s]["mom_rank"] = ranks[s].reindex(sym[s].index)
    return MarketState(idx, btc_bull, breadth, btc_vol_pct, sym)
