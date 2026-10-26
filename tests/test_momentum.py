"""Momentum candidates must be causal: truncating the future cannot change past signals."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.marketstate import build_state
from tycho.momentum import GRIDS, build, liquid_mask, make_expert
from tycho.portfolio import load_universe

DATA_OK = all((Path(__file__).resolve().parent.parent / "data" / f"{s}_1h.parquet").exists() for s in C.UNIVERSE)
CUT = pd.Timestamp("2023-03-01", tz="UTC")


@pytest.mark.skipif(not DATA_OK, reason="historical data not downloaded")
def test_momentum_candidates_are_causal():
    data = {s: df.loc[:"2023-12-31"] for s, df in load_universe(C.UNIVERSE, C.TIMEFRAME).items()}
    cut = {s: df[df.index < CUT] for s, df in data.items()}
    ms_full, ms_cut = build_state(data), build_state(cut)
    for name, grid in GRIDS.items():
        for params, req in grid:
            ex = make_expert(name, params, req)
            _, a, _, _ = build(data, [ex], ms_full, tradable=["BTCUSDT", "SOLUSDT", "DOGEUSDT"])
            _, b, _, _ = build(cut, [ex], ms_cut, tradable=["BTCUSDT", "SOLUSDT", "DOGEUSDT"])
            for key in a:
                k = len(b[key].long_entry)
                for f in a[key].__dataclass_fields__:
                    x, y = getattr(a[key], f)[:k].astype(float), getattr(b[key], f).astype(float)
                    assert np.allclose(x, y, equal_nan=True), (name, params, key, f)


def test_liquidity_gate_is_causal_and_needs_history():
    idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
    df = pd.DataFrame({"qvol": np.r_[np.full(300, 1e6), np.full(100, 1e8)]}, index=idx)
    m = liquid_mask(df)
    assert not m.iloc[:179].any()                      # < 30 days of history: never tradable
    assert m.iloc[-1] and not m.iloc[299]              # turns on only after volume rises
    assert (liquid_mask(df.iloc[:350]).to_numpy() == m.iloc[:350].to_numpy()).all()
