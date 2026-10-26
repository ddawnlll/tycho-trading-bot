"""The live Nautilus plan (latest bar only, trailing windows) must equal the backtest plan row."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.ensemble import build_columns, configured_experts
from tycho.marketstate import build_state
from tycho.nautilus_bt import plan_frames
from tycho.nautilus_live import latest_plan
from tycho.portfolio import load_universe

DATA_OK = all((Path(__file__).resolve().parent.parent / "data" / f"{s}_1h.parquet").exists() for s in C.UNIVERSE)


@pytest.mark.skipif(not DATA_OK, reason="historical data not downloaded")
def test_latest_plan_matches_backtest_plan():
    data = load_universe(C.UNIVERSE, C.TIMEFRAME)
    cols, sigs, _, _ = build_columns(data, configured_experts(), build_state(data, C.REGIME_N))
    full = plan_frames(sigs, cols)
    times = data["BTCUSDT"].index[-3000::250]
    n_true = 0
    for t in times:
        bars = {s: df[df.index <= t].iloc[-(C.REGIME_WARMUP_BARS if s == "BTCUSDT" else C.WARMUP_BARS):]
                for s, df in data.items()}
        plan, state = latest_plan(bars)
        assert set(state["experts"]) == {e[0] for e in C.EXPERTS}
        for k, row in plan.items():
            ref = full[k].loc[t.value]
            for f in ("long_entry", "long_exit"):
                assert bool(row[f].iloc[0]) == bool(ref[f]), (k, t, f)
                n_true += bool(ref[f])
            for f in ("long_stop", "long_trail"):
                a, b = row[f].iloc[0], ref[f]
                assert (np.isnan(a) and np.isnan(b)) or abs(a / b - 1) < 1e-6, (k, t, f, a, b)
    assert n_true > 0  # at least one entry/exit signal was actually compared
