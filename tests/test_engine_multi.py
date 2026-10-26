import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_engine import _rand_df
from tycho.engine import Costs, run
from tycho.engine_multi import run_multi
from tycho.strategies import donchian, trend_pullback


def test_single_symbol_parity_with_engine():
    for seed in (0, 1, 2):
        df = _rand_df(1500, seed).assign(funding=0.0001)
        for sig in (donchian(df, 20, 10, 2.0), trend_pullback(df)):
            a = run(df, sig, risk=0.01, max_lev=3, costs=Costs())
            b = run_multi({"X": df}, {"X": sig}, risk=0.01, max_lev_pos=3, max_gross=1e9,
                          max_positions=99, costs=Costs())
            assert len(a["trades"]) == len(b["trades"]) > 0
            assert np.allclose(a["equity"].to_numpy(), b["equity"].to_numpy(), rtol=1e-9)


def test_caps_respected_and_rank_order():
    dfs = {f"S{k}": _rand_df(800, k) for k in range(5)}
    sigs = {s: donchian(d, 10, 5, 2.0, allow_short=False) for s, d in dfs.items()}
    res = run_multi(dfs, sigs, risk=0.05, max_lev_pos=1.0, max_gross=2.0, max_positions=2)
    t = res["trades"]
    # never more than 2 concurrent positions
    ev = pd.concat([pd.Series(1, t.entry_time), pd.Series(-1, t.exit_time)]).sort_index(kind="stable")
    # exits at same timestamp happen before entries in the engine
    ev = ev.groupby(level=0).apply(lambda x: x.sort_values()).reset_index(level=0, drop=True)
    assert ev.cumsum().max() <= 2
    # rank: two identical symbols, one slot -> only the higher-ranked one ever trades
    d = _rand_df(800, 3)
    sig = donchian(d, 10, 5, 2.0, allow_short=False)
    rank = {"LO": np.zeros(len(d)), "HI": np.ones(len(d))}
    res2 = run_multi({"LO": d, "HI": d}, {"LO": sig, "HI": sig}, rank=rank, risk=0.05,
                     max_lev_pos=1.0, max_gross=1.0, max_positions=1)
    assert len(res2["trades"]) > 0 and set(res2["trades"].symbol) == {"HI"}


def test_groups_allow_one_expert_per_coin():
    d = _rand_df(800, 4)
    sig = donchian(d, 10, 5, 2.0, allow_short=False)
    res = run_multi({"A:X": d, "B:X": d}, {"A:X": sig, "B:X": sig}, groups={"A:X": "X", "B:X": "X"},
                    risk=0.01, max_lev_pos=1.0, max_gross=5.0, max_positions=10)
    assert len(res["trades"]) > 0 and set(res["trades"].symbol) == {"A:X"}  # B never co-holds X
    free = run_multi({"A:X": d, "B:X": d}, {"A:X": sig, "B:X": sig},
                     risk=0.01, max_lev_pos=1.0, max_gross=5.0, max_positions=10)
    assert set(free["trades"].symbol) == {"A:X", "B:X"}
