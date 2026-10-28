"""Ported from ddawnlll/tycho @ 0a309c4, tests/test_nulls.py — imports and fixture pruning.

Only the tests of the null *generators* are ported, unchanged in their assertions. The
source tests `test_zero_eligible_columns_on_multi_asset_tape`,
`test_buy_and_hold_uses_same_pipeline_with_score_plus_one`,
`test_btc_baseline_recomputes_b_with_n_equals_one`, `test_no_trade_keeps_equity_constant`
and `test_baseline_and_null_trials_carry_their_role` are not ported: they exercise the
E-001 ledger (`e001.build_experiment`, `e001._simulate`, `trials`) and the E-001 synthetic
tape, none of which exists in v2 — G0-T1 copies the null generators only, see
`tycho/v2/nulls.py`.
"""

import numpy as np
import pandas as pd
import pytest

from tycho.v2 import nulls


def _fixture(values, eligible=None):
    length = len(next(iter(values.values()))) if isinstance(values, dict) else len(values)
    index = pd.date_range("2020-01-01", periods=length, freq="D")
    w = pd.DataFrame(values, index=index, dtype=float)
    if eligible is None:
        eligible = True
    eligible = pd.DataFrame(eligible, index=w.index, columns=w.columns, dtype=bool)
    return w, eligible


def _lengths(w, eligible, sign, asset="A"):
    return sorted(
        block["length"] for block in nulls.blocks(w, eligible)
        if block["asset"] == asset and (block["sign"] != 0) == (sign != 0)
    )


def test_scramble_preserves_abs_weights_and_date_gross():
    w, eligible = _fixture({"A": [1.0, -2.0, 0.0, 3.0, -4.0],
                            "B": [0.5, 0.5, -1.5, 0.0, 2.0]})
    original_gross = w.abs().sum(axis=1)
    assert original_gross.nunique() > 1
    for item in nulls.scramble(w, eligible, reps=16, seed=11):
        null = item["weights"]
        pd.testing.assert_frame_equal(null.abs(), w.abs())
        pd.testing.assert_series_equal(null.abs().sum(axis=1), original_gross)
        assert item["blocks"]


def test_scramble_blocks_can_merge_into_one_run():
    w, eligible = _fixture({"A": [1.0, 1.0, -1.0, -1.0]})
    merged = []
    for item in nulls.scramble(w, eligible, reps=32, seed=0):
        merged.extend(
            block["length"] for block in nulls.blocks(item["weights"], eligible)
            if block["sign"] != 0
        )
    assert 4 in merged


def test_random_entry_preserves_run_and_flat_length_multisets_and_eligibility():
    eligible = {"A": [True, True, False, True, True, True, False, True]}
    w, eligible = _fixture({"A": [1.0, 2.0, 0.0, 0.0, -3.0, -3.5, 0.0, 4.0]}, eligible)
    original_runs = _lengths(w, eligible, 1)
    original_flats = _lengths(w, eligible, 0)
    original_total = float(w.abs().sum().iloc[0])
    assert original_runs and original_flats
    for item in nulls.random_entry(w, eligible, reps=16, seed=5, gross_cap=100.0):
        null = item["weights"]
        assert _lengths(null, eligible, 1) == original_runs
        assert _lengths(null, eligible, 0) == original_flats
        assert item["pre_cap_total_abs"]["A"] == pytest.approx(original_total)
        assert (null.to_numpy()[~eligible.to_numpy()] == 0.0).all()


def test_gross_cap_shrinks_proportionally_and_reports_bind_count():
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    w = pd.DataFrame({"A": [1.5, 2.0, 1.0], "B": [1.0, 1.5, 0.5],
                      "C": [0.5, 0.0, 0.6]}, index=index)
    pre_gross = w.abs().sum(axis=1)
    assert pre_gross.nunique() == 3 and (pre_gross > 2.0).all()
    capped, binds = nulls.apply_gross_cap(w, 2.0)
    assert binds == 3
    assert np.allclose(capped.abs().sum(axis=1), 2.0)
    ratio = (capped / w).replace([np.inf, -np.inf], np.nan)
    for date in index:
        nonzero = ratio.loc[date].dropna()
        assert np.allclose(nonzero, nonzero.iloc[0])

    _, eligible = _fixture({"A": [1.5, 2.0, 1.0], "B": [1.0, 1.5, 1.5]})
    for item in nulls.random_entry(
        pd.DataFrame({"A": [1.5, 2.0, 1.0], "B": [1.0, 1.5, 1.5]}, index=index),
        eligible, reps=2, seed=1, gross_cap=2.0,
    ):
        assert (item["weights"].abs().sum(axis=1) <= 2.0 + 1e-12).all()
        assert item["cap_bind_days"] == 3


def test_multi_asset_random_entry_has_non_constant_gross_before_cap():
    w, eligible = _fixture({
        "A": [1.5, 2.0, 1.0, 0.0],
        "B": [1.0, 1.5, 0.5, -2.0],
        "C": [0.5, 0.0, 0.6, 0.0],
    })
    assert w.abs().sum(axis=1).nunique() > 1
    for item in nulls.random_entry(w, eligible, reps=4, seed=3, gross_cap=2.0):
        assert (item["weights"].abs().sum(axis=1) <= 2.0 + 1e-12).all()
        assert set(item["pre_cap_total_abs"]) == {"A", "B", "C"}


def test_same_seed_gives_the_same_null_panels():
    w, eligible = _fixture({"A": [1.0, -1.0, 0.0, 2.0], "B": [0.0, 1.0, 1.0, -1.0]})
    for make in (
        lambda seed: nulls.scramble(w, eligible, reps=3, seed=seed),
        lambda seed: nulls.random_entry(w, eligible, reps=3, seed=seed),
    ):
        first = list(make(19))
        second = list(make(19))
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a["weights"], b["weights"])
