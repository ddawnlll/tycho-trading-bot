"""Causal per-bar features for the meta-label model (value at i uses bars <= i only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import atr, ema


def bar_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    a = atr(df)
    f = pd.DataFrame(index=df.index)
    for n in (6, 24, 72, 180):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["atr_pct"] = a / c
    f["atr_ratio"] = a / a.rolling(180).mean()
    for n in (21, 50, 200):
        f[f"dist_ema{n}"] = (c - ema(c, n)) / a
    f["slope_ema50"] = ema(c, 50).pct_change(10)
    f["vol_z"] = (df["volume"] - df["volume"].rolling(180).mean()) / df["volume"].rolling(180).std()
    f["funding_3d"] = df["funding"].rolling(18).sum()
    f["range_pos_40"] = (c - df["low"].rolling(40).min()) / (df["high"].rolling(40).max() - df["low"].rolling(40).min())
    bc = btc["close"].reindex(df.index).ffill()
    f["btc_ret_180"] = np.log(bc / bc.shift(180))
    f["btc_dist_ema1200"] = bc / ema(bc, 1200) - 1
    f["rel_strength_180"] = f["ret_180"] - f["btc_ret_180"]
    f["hour"] = df.index.hour
    return f
