"""Compare the v2 HAC implementation with statsmodels on five seeded series."""

import numpy as np
import pytest
import statsmodels.api as sm

from tycho.v2.stats import beta_adjusted_alpha


@pytest.mark.parametrize("seed", range(5))
def test_hac_matches_statsmodels(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, 1000)
    y = 0.0005 + 0.6 * x + rng.normal(0, 0.003, len(x))
    # Both implementations: maxlags=5, Bartlett kernel, use_correction=True.
    actual = beta_adjusted_alpha(y, x, hac_maxlags=5)
    reference = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5, "kernel": "bartlett", "use_correction": True},
    )
    np.testing.assert_allclose(
        [actual["alpha"], actual["beta"], actual["alpha_se"], actual["beta_se"]],
        [reference.params[0], reference.params[1], reference.bse[0], reference.bse[1]],
        rtol=1e-8, atol=0,
    )
