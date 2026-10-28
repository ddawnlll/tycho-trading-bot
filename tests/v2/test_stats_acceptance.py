"""G0-T1 acceptance tests for `tycho/v2/stats.py`.

Acceptance criteria implemented here (docs/v2/G0_TASKS.md, G0-T1):
  * with one trial and zero trial-Sharpe variance, the DSR equals the probabilistic Sharpe
    ratio against 0;
  * the DSR decreases monotonically as the number of trials grows, N = 1, 10, 100, 1000,
    for a fixed return series;
  * `beta_adjusted_alpha` recovers a planted alpha and beta within three standard errors;
  * `hansen_spa` rejects 20 pure-noise strategies at p <= 0.05 in at most ~5% of 200 seeds
    (see `tests/v2/test_spa_acceptance.py` for the rate itself).
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from tycho.v2.stats import beta_adjusted_alpha, deflated_sharpe

# A fixed "return series": per-period Sharpe 0.15, 1500 observations, mildly non-normal.
SR = 0.15
NOBS = 1500
SKEW = -0.4
KURTOSIS = 5.0


def probabilistic_sharpe_ratio(sr, n_obs, skew, kurtosis, benchmark_sr=0.0):
    """PSR from the Bailey & Lopez de Prado definition, written out independently."""
    variance = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    scale = math.sqrt(n_obs - 1) / math.sqrt(variance)
    return float(norm.cdf((sr - benchmark_sr) * scale))


def test_one_trial_zero_variance_dsr_equals_psr_against_zero():
    result = deflated_sharpe(SR, NOBS, SKEW, KURTOSIS, 1, 0.0)
    expected = probabilistic_sharpe_ratio(SR, NOBS, SKEW, KURTOSIS, 0.0)
    assert result["sr0"] == 0.0
    assert result["dsr"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert result["dsr"] == result["psr_vs_zero"]
    # And the PSR is the zero-benchmark special case of the same formula, not a no-op.
    assert expected == pytest.approx(0.9999, abs=1e-3)


@pytest.mark.parametrize("n_trials", [1, 10, 100, 1000])
def test_dsr_equals_psr_for_one_trial_only(n_trials):
    result = deflated_sharpe(SR, NOBS, SKEW, KURTOSIS, n_trials, 0.002)
    psr = probabilistic_sharpe_ratio(SR, NOBS, SKEW, KURTOSIS, 0.0)
    assert result["psr_vs_zero"] == pytest.approx(psr, rel=0, abs=1e-12)
    if n_trials == 1:
        assert result["dsr"] == pytest.approx(psr, rel=0, abs=1e-12)
    else:
        assert result["dsr"] < psr


def test_dsr_decreases_monotonically_in_number_of_trials():
    dsr = [deflated_sharpe(SR, NOBS, SKEW, KURTOSIS, n, 0.002)["dsr"] for n in (1, 10, 100, 1000)]
    sr0 = [deflated_sharpe(SR, NOBS, SKEW, KURTOSIS, n, 0.002)["sr0"] for n in (1, 10, 100, 1000)]
    assert dsr == sorted(dsr, reverse=True)
    assert len(set(dsr)) == len(dsr)  # strictly decreasing, not merely non-increasing
    assert sr0 == sorted(sr0)
    assert len(set(sr0)) == len(sr0)


def test_dsr_monotone_in_trial_variance():
    dsr = [deflated_sharpe(SR, NOBS, SKEW, KURTOSIS, 100, v)["dsr"] for v in (0.0, 0.0005, 0.002, 0.01)]
    assert dsr == sorted(dsr, reverse=True)


PLANTED_ALPHA = 0.0008
PLANTED_BETA = 0.6


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_alpha_recovers_planted_alpha_and_beta_within_three_standard_errors(seed):
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0, 0.01, 4000)
    strategy = PLANTED_ALPHA + PLANTED_BETA * benchmark + rng.normal(0.0, 0.003, 4000)
    result = beta_adjusted_alpha(strategy, benchmark, hac_maxlags=10)
    assert result["alpha"] == pytest.approx(PLANTED_ALPHA, abs=3.0 * result["alpha_se"])
    assert result["beta"] == pytest.approx(PLANTED_BETA, abs=3.0 * result["beta_se"])
    assert result["alpha_se"] > 0.0 and result["beta_se"] > 0.0
    assert result["t"] == pytest.approx(result["alpha"] / result["alpha_se"], rel=1e-12)
    assert result["p_one_sided"] == pytest.approx(float(norm.sf(result["t"])), rel=0, abs=1e-15)


def test_alpha_with_zero_noise_is_recovered_exactly_and_gets_a_null_standard_error():
    rng = np.random.default_rng(5)
    benchmark = rng.normal(0.0, 0.01, 500)
    strategy = 0.001 + 0.5 * benchmark  # perfect fit: residual variance is 0
    result = beta_adjusted_alpha(strategy, benchmark, hac_maxlags=3)
    assert result["alpha"] == pytest.approx(0.001, abs=1e-12)
    assert result["beta"] == pytest.approx(0.5, abs=1e-12)
    # A zero-residual fit has a degenerate HAC covariance; the standard error must collapse
    # with it instead of being replaced by a made-up positive number.
    assert result["alpha_se"] <= 1e-12
    assert result["beta_se"] <= 1e-12
    assert math.isnan(result["p_one_sided"])
    assert result["pass_90"] is False
