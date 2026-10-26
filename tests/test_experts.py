"""MarketState and every expert must be causal: truncating the future can't change the past."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.experts import CATALOG
from tycho.marketstate import build_state
from tycho.portfolio import load_universe

DATA_OK = all((Path(__file__).resolve().parent.parent / "data" / f"{s}_1h.parquet").exists() for s in C.UNIVERSE)
CUT = pd.Timestamp("2023-06-01", tz="UTC")


@pytest.fixture(scope="module")
def states():
    data = {s: df.loc[:"2023-12-31"] for s, df in load_universe(C.UNIVERSE, C.TIMEFRAME).items()}
    cut = {s: df[df.index < CUT] for s, df in data.items()}
    return data, build_state(data), cut, build_state(cut)


@pytest.mark.skipif(not DATA_OK, reason="historical data not downloaded")
def test_marketstate_is_causal(states):
    data, full, cut, part = states
    m = full.index < CUT
    for name in ("btc_bull", "breadth", "btc_vol_pct"):
        a, b = getattr(full, name), getattr(part, name)
        a = a[a.index < CUT].astype(float)
        assert np.allclose(a.to_numpy(), b.astype(float).reindex(a.index).to_numpy(), equal_nan=True), name
    for s in C.UNIVERSE:
        a, b = full.sym[s][full.sym[s].index < CUT].astype(float), part.sym[s].astype(float)
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), s
    assert m.sum() > 1000


@pytest.mark.skipif(not DATA_OK, reason="historical data not downloaded")
def test_experts_are_causal(states):
    data, full, cut, part = states
    for name, ex in CATALOG.items():
        for s in ("BTCUSDT", "SOLUSDT", "DOGEUSDT"):
            k = len(cut[s])
            a = ex.signals(data[s], full.for_symbol(s))
            b = ex.signals(cut[s], part.for_symbol(s))
            for f in a.__dataclass_fields__:
                assert np.allclose(getattr(a, f)[:k].astype(float), getattr(b, f).astype(float), equal_nan=True), (name, s, f)
