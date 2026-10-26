"""NautilusTrader order handling of EnsembleStrategy on synthetic data (simulated exchange)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nautilus_trader.model.instruments import CryptoPerpetual

from tycho import nautilus_bt as NB
from tycho.engine import Signals

INFO_OK = NB.INFO_PATH.exists()


def _data(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="4h", tz="UTC")
    c = np.array(prices, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c) + 1, low=np.minimum(o, c) - 1, close=c, volume=10.0,
                             funding=0.0), index=idx)


def _sig(n, entry_at, stop, trail=None):
    le = np.zeros(n, bool)
    le[entry_at] = True
    nan = np.full(n, np.nan)
    tr = nan.copy() if trail is None else np.asarray(trail, float)
    return Signals(le, np.zeros(n, bool), np.full(n, stop), nan, tr, nan, np.zeros(n, bool), np.zeros(n, bool))


@pytest.mark.skipif(not INFO_OK, reason="data/exchange_info.json missing")
def test_trailing_stop_is_replaced_not_modified_then_fills():
    prices = [100 + 2 * i for i in range(20)] + [130 - 5 * i for i in range(10)]  # up, then down through the trail
    df = _data(prices)
    n = len(df)
    trail = np.r_[np.full(5, np.nan), df.close.to_numpy()[5:] - 8]
    data = {"BTCUSDT": df}
    cols = {"trend_long:BTCUSDT": df}
    sigs = {"trend_long:BTCUSDT": _sig(n, 3, 90.0, trail)}
    res = NB.run(data, cols, sigs, df.index[0], df.index[-1], risk=0.01, max_positions=5, max_gross=2.0, max_lev_pos=1.0)
    o = res["orders"]
    stops = o[o["type"] == "STOP_MARKET"]
    assert len(res["positions"]) == 1 and pd.notna(res["positions"]["ts_closed"].iloc[0])
    assert len(stops) > 3                                     # stop was re-placed as the trail rose
    assert set(stops["status"]) <= {"CANCELED", "FILLED"} and (stops["status"] == "FILLED").sum() == 1
    assert res["open_orders"] == 0 and res["held_end"] == {}  # nothing dangling, coin released
    closed_px = float(res["positions"]["avg_px_close"].iloc[0])
    assert closed_px > 100                                    # exited on the raised trail, not the initial 90 stop


@pytest.mark.skipif(not INFO_OK, reason="data/exchange_info.json missing")
def test_denied_entry_frees_the_slot(monkeypatch):
    orig = NB.make_instrument

    def tiny_max_qty(symbol, filt, **kw):  # RiskEngine denies any order above max_quantity
        d = CryptoPerpetual.to_dict(orig(symbol, filt, **kw))
        d["max_quantity"] = filt["step"]
        return CryptoPerpetual.from_dict(d)
    monkeypatch.setattr(NB, "make_instrument", tiny_max_qty)
    df = _data([100 + i for i in range(15)])
    n = len(df)
    le = np.zeros(n, bool)
    le[[2, 6, 10]] = True
    nan = np.full(n, np.nan)
    sig = Signals(le, np.zeros(n, bool), np.full(n, 90.0), nan, nan, nan, np.zeros(n, bool), np.zeros(n, bool))
    cols = {"trend_long:BTCUSDT": df}
    res = NB.run({"BTCUSDT": df}, cols, {"trend_long:BTCUSDT": sig}, df.index[0], df.index[-1],
                 risk=0.01, max_positions=1, max_gross=2.0, max_lev_pos=1.0)
    o = res["orders"]
    assert (o["status"] == "DENIED").sum() == 3               # every signal was attempted again: slot was freed
    assert res["held_end"] == {} and len(res["positions"]) == 0
