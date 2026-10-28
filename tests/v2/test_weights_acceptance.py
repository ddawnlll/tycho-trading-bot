"""G0-T1 acceptance test: `target_weights` caps on random inputs (property test, 200 cases).

The gross cap and the per-asset share of the volatility cap must both hold on arbitrary
admissible inputs, not only on the hand-built cases in `test_weights_port.py`.
"""

import numpy as np
import pandas as pd
import pytest

from tycho.v2.weights import risk_budget, target_weights

CASES = 200


def _case(case):
    """Deterministic pseudo-random admissible sizing problem for one property case."""
    rng = np.random.default_rng(10_000 + case)
    n = int(rng.integers(1, 31))
    ids = [f"A{i:02d}" for i in range(n)]

    vols = rng.uniform(0.1, 1.5, n)
    factor = rng.standard_normal((n, max(2, n // 2)))
    corr = factor @ factor.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr * np.outer(vols, vols), index=ids, columns=ids)

    scores = pd.Series(rng.uniform(-1.0, 1.0, n), index=ids)
    # A few abstentions and degenerate vols: the function must zero them, not shrink.
    abstain = rng.random(n) < 0.15
    scores[abstain] = np.nan
    vols_series = pd.Series(vols, index=ids)
    vols_series[rng.random(n) < 0.1] = 0.0

    params = {
        "target_vol": float(rng.uniform(0.05, 0.40)),
        "rho": float(rng.uniform(0.05, 0.95)),
        "vol_cap_multiple": float(rng.uniform(1.0, 3.0)),
        "gross_cap": float(rng.uniform(0.5, 3.0)),
    }
    return scores, vols_series, cov, params, n


@pytest.mark.parametrize("case", range(CASES))
def test_gross_and_per_asset_vol_caps_hold(case):
    scores, vols, cov, params, n = _case(case)
    weights, diag = target_weights(
        scores, vols, cov,
        params["target_vol"], params["rho"], n,
        params["vol_cap_multiple"], params["gross_cap"],
        on_abstain="flat",
    )
    b = risk_budget(params["target_vol"], n, params["rho"])
    assert diag["b"] == pytest.approx(b, rel=1e-15)

    # Gross cap holds after the shrink.
    assert weights.abs().sum() <= params["gross_cap"] + 1e-12
    # The shrink never levers up.
    assert 0.0 < diag["s"] <= 1.0

    # Per-asset cap: |w_i| <= score_i * b / vol_i <= b / vol_i, and the per-asset
    # volatility contribution |w_i| * vol_i therefore never exceeds the risk budget b.
    for asset in weights.index:
        vol = vols[asset]
        weight = weights[asset]
        if not np.isfinite(vol) or vol <= 0.0 or not np.isfinite(scores[asset]):
            assert weight == 0.0
            continue
        assert abs(weight) <= b / vol + 1e-12
        assert abs(weight) * vol <= b + 1e-12

    # The binding cap is reported truthfully and the book stays inside the vol cap.
    if diag["cap"] == "gross_cap":
        assert weights.abs().sum() == pytest.approx(params["gross_cap"], abs=1e-12)
    if diag["cap"] == "vol_cap":
        assert diag["forecast_vol"] * diag["s"] <= (
            params["vol_cap_multiple"] * params["target_vol"] + 1e-12
        )
    if diag["forecast_vol"] > 0.0:
        assert diag["forecast_vol"] * diag["s"] <= (
            params["vol_cap_multiple"] * params["target_vol"] + 1e-12
        )
