"""G0-T1 acceptance test: Hansen SPA on pure noise must not reject above its size.

Twenty pure-noise strategies (iid, zero mean, 1000 observations) are tested against a zero
benchmark over 200 seeds. Hansen's consistent p-value is the
headline; the test requires the rejection rate at p <= 0.05 to stay at or below 5%.
"""

import numpy as np
import pytest

from tycho.v2.stats import hansen_spa, romano_wolf_stepm

SEEDS = 200
N_STRATEGIES = 20
N_OBS = 1000
REPS = 500
BLOCK_SIZE = 10
ALPHA = 0.05


def _noise(seed):
    return np.random.default_rng(seed).standard_normal((N_OBS, N_STRATEGIES)) * 0.01


def _rejection_rate():
    rejections = [
        hansen_spa(
            _noise(seed), reps=REPS, block_size=BLOCK_SIZE, seed=50_000 + seed, alpha=ALPHA
        )["reject"]
        for seed in range(SEEDS)
    ]
    return float(np.mean(rejections))


def test_spa_on_pure_noise_rejects_at_most_five_percent():
    rate = _rejection_rate()
    assert rate <= 0.05, f"SPA false-positive rate {rate:.3%} over {SEEDS} seeds exceeds 5%"


def test_spa_power_for_annual_sharpe_two():
    # Daily iid returns, one edge among 19 nulls; independent bootstrap seeds.
    detections = 0
    for seed in range(SEEDS):
        returns = _noise(seed)
        returns[:, 0] += 0.01 * 2.0 / np.sqrt(252)
        result = hansen_spa(returns, reps=REPS, block_size=BLOCK_SIZE,
                            seed=50_000 + seed, alpha=ALPHA)
        detections += result["p_consistent"] <= ALPHA and 0 in result["better_models"]
    rate = detections / SEEDS
    print(f"SPA power: {detections}/{SEEDS} = {rate:.1%}")
    assert rate >= 0.80


def test_spa_reports_all_three_pvalues_and_is_reproducible():
    noise = _noise(3)
    first = hansen_spa(noise, reps=200, block_size=BLOCK_SIZE, seed=7)
    again = hansen_spa(noise, reps=200, block_size=BLOCK_SIZE, seed=7)
    assert first["p_consistent"] == again["p_consistent"]
    assert 0.0 <= first["p_lower"] <= 1.0
    assert 0.0 <= first["p_consistent"] <= 1.0
    assert 0.0 <= first["p_upper"] <= 1.0
    assert first["n_models"] == N_STRATEGIES and first["n_obs"] == N_OBS
    assert first["reject"] == (first["p_consistent"] <= ALPHA)


def test_spa_detects_one_planted_signal_among_noise():
    noise = _noise(11)
    noise[:, 3] += 0.004  # 0.4% per period on a 1% volatility, ~4 sigma over 1000 obs
    result = hansen_spa(noise, reps=REPS, block_size=BLOCK_SIZE, seed=1)
    assert result["p_consistent"] <= ALPHA
    assert result["reject"] is True
    assert result["better_models"] == [3]


def test_stepm_selects_only_the_planted_signal():
    noise = _noise(11)
    noise[:, 3] += 0.004
    noise[:, 9] += 0.004
    result = romano_wolf_stepm(noise, reps=REPS, block_size=BLOCK_SIZE, seed=2)
    assert result["superior_models"] == [3, 9]
    assert result["n_superior"] == 2

    # On pure noise the stepwise procedure selects nothing at this size.
    quiet = romano_wolf_stepm(_noise(2), reps=REPS, block_size=BLOCK_SIZE, seed=3)
    assert quiet["superior_models"] == []


def test_wrappers_reject_bad_input():
    with pytest.raises(ValueError, match="same number of rows"):
        hansen_spa(_noise(0), benchmark=np.zeros(5), reps=10, seed=0)
    bad = _noise(0)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        hansen_spa(bad, reps=10, seed=0)


def test_single_strategy_benchmark_is_respected():
    # One strategy with a tiny positive mean: with a zero benchmark it fails to reject at
    # this sample size, with a benchmark that already earns more it cannot reject either.
    returns = _noise(4)[:, :1] + 0.0001
    zero_benchmark = hansen_spa(returns, reps=REPS, block_size=BLOCK_SIZE, seed=5)
    ahead = hansen_spa(returns, benchmark=np.full(N_OBS, 0.005), reps=REPS, block_size=BLOCK_SIZE, seed=5)
    assert zero_benchmark["n_models"] == 1
    assert ahead["reject"] is False
