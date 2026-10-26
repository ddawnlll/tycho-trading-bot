"""In-venue funding must charge exactly qty * bar open * rate while a position is held."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import nautilus_bt as NB
from tycho import nautilus_funding as NF
from tycho.engine import Costs, Signals
from tycho.engine_multi import run_multi


@pytest.mark.skipif(not NB.INFO_PATH.exists(), reason="data/exchange_info.json missing")
def test_funding_charged_in_venue_matches_tycho():
    n = 40
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    c = np.full(n, 100.0)
    df = pd.DataFrame(dict(open=c, high=c + 1, low=c - 1, close=c, volume=10.0, funding=0.0), index=idx)
    df.loc[df.index.hour == 0, "funding"] = 0.001  # one funding event per day, 10 bp
    le = np.zeros(n, bool)
    le[2] = True
    nan = np.full(n, np.nan)
    sig = Signals(le, np.zeros(n, bool), np.full(n, 50.0), nan, nan, nan, np.zeros(n, bool), np.zeros(n, bool))
    filt = NB.exchange_filters()
    kw = dict(risk=0.01, max_positions=1, max_gross=2.0, max_lev_pos=1.0)
    nt = NF.run({"BTCUSDT": df}, {"x:BTCUSDT": df}, {"x:BTCUSDT": sig}, idx[0], idx[-1], filt,
                {"BTCUSDT": 0.0}, **kw)
    ty = run_multi({"x:BTCUSDT": df}, {"x:BTCUSDT": sig}, costs=Costs(fee=0.0, slippage=0.0), **kw)
    qty = float(nt["positions"]["peak_qty"].iloc[0])
    held = df.index[df.index >= nt["positions"]["ts_opened"].iloc[0]]
    expected = qty * 100.0 * 0.001 * int((held.hour == 0).sum())
    assert expected > 0 and abs(nt["funding_paid"] - expected) < 1e-6
    assert abs(nt["equity"].iloc[-1] / ty["equity"].iloc[-2] - 1) < 2e-3   # same drag in both engines
