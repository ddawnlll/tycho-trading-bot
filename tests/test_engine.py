import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho.engine import Costs, Signals, run
from tycho.strategies import donchian, ma_trend, trend_pullback


def _df(o, h, l, c):
    idx = pd.date_range("2024-01-01", periods=len(o), freq="4h", tz="UTC")
    return pd.DataFrame(dict(open=o, high=h, low=l, close=c, funding=0.0), index=idx)


def _sig(n, long_at=None, stop=None):
    z = np.zeros(n, bool)
    le = z.copy()
    if long_at is not None:
        le[long_at] = True
    nan = np.full(n, np.nan)
    st = np.full(n, stop if stop is not None else np.nan)
    return Signals(le, z.copy(), st, nan, nan, nan, z.copy(), z.copy())


def test_entry_fills_next_open_and_stop_at_price():
    df = _df([100, 101, 102, 100], [100, 102, 103, 100], [100, 100, 99, 94], [100, 101, 102, 95])
    res = run(df, _sig(4, long_at=0, stop=95.0), risk=0.01, max_lev=10, costs=Costs(0, 0))
    t = res["trades"].iloc[0]
    assert t.entry == 101 and t.reason == "stop" and t.exit == 95.0
    assert abs(t.r + 1.0) < 1e-9  # lost exactly 1R without costs
    assert abs(res["equity"].iloc[-1] - 9_900) < 1e-6


def test_gap_through_stop_fills_at_open():
    df = _df([100, 101, 90], [100, 102, 91], [100, 100, 89], [100, 101, 90])
    res = run(df, _sig(3, long_at=0, stop=95.0), risk=0.01, max_lev=10, costs=Costs(0, 0))
    t = res["trades"].iloc[0]
    assert t.exit == 90 and t.r < -1.5


def test_fees_reduce_equity():
    df = _df([100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100])
    res = run(df, _sig(3, long_at=0, stop=90.0), risk=0.01, max_lev=10, costs=Costs(0.0005, 0.0))
    assert res["equity"].iloc[-1] < 10_000


def _rand_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    return _df(o, h, l, c)


def test_signals_are_causal():
    df = _rand_df()
    k = 400
    for fn in (trend_pullback, donchian, ma_trend):
        full, cut = fn(df), fn(df.iloc[:k])
        for name in ("long_entry", "short_entry", "long_stop", "short_stop", "long_trail", "short_trail"):
            a, b = getattr(full, name)[:k], getattr(cut, name)
            assert np.allclose(a, b, equal_nan=True), (fn.__name__, name)


def test_liquidation_books_once_and_stops():
    df = _df([100, 100, 40, 200], [100, 100, 40, 200], [100, 100, 40, 200], [100, 100, 40, 200])
    res = run(df, _sig(4, long_at=0, stop=1e-9), risk=5.0, max_lev=2.0, costs=Costs(0, 0))
    assert list(res["trades"].reason) == ["liquidated"]
    assert res["equity"].iloc[-1] == 0.0
