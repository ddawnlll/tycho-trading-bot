"""v2 research statistics: deflated Sharpe, block bootstrap Sharpe, HAC alpha, SPA.

Copied from `ddawnlll/tycho` at commit `0a309c4`, file `src/tycho/stats.py`
(`_sharpe_variance`, `deflated_sharpe`, `mean_holding_period`, `_annualized_sharpe`,
`block_bootstrap_sharpe`, `beta_adjusted_alpha`). Two deliberate differences, both forced
by this repository's dependency policy:

1. `beta_adjusted_alpha` uses an explicit Newey-West HAC covariance in numpy instead of
   `statsmodels.OLS(cov_type="HAC")`, because the `v2` optional group only adds `arch` and
   `scipy` (`docs/v2/G0_TASKS.md`, G0-T1). The estimator is the same: Bartlett kernel with
   weights ``1 - lag/(maxlags+1)`` and the ``n/(n-k)`` small-sample correction, so the
   point estimates, standard errors and one-sided p-value keep the source meaning.
   `alpha_se` and `beta_se` are returned in addition to the source keys.
2. `PendingParameterError` is defined here instead of being imported from
   `tycho.config`, because that module in this repository is unrelated (frozen v1 config)
   and v2 must not depend on it.

`arch` is an optional dependency: the two multiple-comparison wrappers import it lazily,
so the rest of this module works with the base dependencies only.
"""

import math

import numpy as np

_EULER_GAMMA = 0.5772156649015329  # Bailey & Lopez de Prado (2014)
_DAYS_PER_YEAR = 365
_ALPHA_LEVEL = 0.10  # one-sided 90% for the beta-adjusted alpha gate


class PendingParameterError(RuntimeError):
    """A parameter that the research protocol leaves PENDING was used in a calculation."""


def _require(value, name):
    # Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::_require.
    # None is what an unset config entry looks like after load(); both mean PENDING.
    if value is None or (isinstance(value, str) and value in ("PENDING", "")):
        raise PendingParameterError(f"{name} is PENDING")
    return value


def _arch(name):
    """Lazily import one `arch.bootstrap` class with an actionable error message."""
    try:
        from arch import bootstrap
    except ImportError as exc:  # pragma: no cover - exercised only without the v2 extra
        raise ImportError(
            "arch is required for the v2 statistics; install with: pip install -e '.[v2]'"
        ) from exc
    return getattr(bootstrap, name)


def _sharpe_variance(sr, skew, kurtosis):
    """Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::_sharpe_variance."""
    variance = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if variance <= 0:
        raise ValueError("skew/kurtosis make the Sharpe variance term non-positive")
    return variance


def deflated_sharpe(sr, n_obs, skew, kurtosis, n_trials_eff, trial_sr_variance):
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::deflated_sharpe.

    sr and SR0 are per-period (non-annualized) Sharpe; kurtosis is non-excess
    (normal = 3). n_trials_eff and trial_sr_variance are PENDING in the
    monograph, so they are required arguments.
    """
    from scipy.stats import norm

    n_eff = _require(n_trials_eff, "dsr_trial_effective_n")
    var = _require(trial_sr_variance, "dsr_trial_sharpe_variance")
    if n_eff < 1:
        raise ValueError("n_trials_eff must be >= 1")
    scale = math.sqrt(n_obs - 1) / math.sqrt(_sharpe_variance(sr, skew, kurtosis))
    psr_vs_zero = float(norm.cdf(sr * scale))
    if n_eff == 1:
        # Expected max Sharpe over one trial is 0, so DSR is just the PSR.
        return {"dsr": psr_vs_zero, "sr0": 0.0, "psr_vs_zero": psr_vs_zero}
    sr0 = math.sqrt(var) * (
        (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_eff)
        + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_eff * math.e))
    )
    return {
        "dsr": float(norm.cdf((sr - sr0) * scale)),
        "sr0": float(sr0),
        "psr_vs_zero": psr_vs_zero,
    }


def mean_holding_period(w):
    """Mean length in days of nonzero same-sign runs, pooled across assets.

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::mean_holding_period.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    lengths = []
    for column in w.T:
        run = 0
        previous = 0.0
        for value in column:
            sign = float(np.sign(value))
            if sign != 0.0 and sign == previous:
                run += 1
            elif sign != 0.0:
                if run:
                    lengths.append(run)
                run = 1
            else:
                if run:
                    lengths.append(run)
                run = 0
            previous = sign
        if run:
            lengths.append(run)
    return float(np.mean(lengths)) if lengths else 0.0


def _annualized_sharpe(values):
    """Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::_annualized_sharpe."""
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(values.mean() / sd * math.sqrt(_DAYS_PER_YEAR))


def block_bootstrap_sharpe(pnl, mean_block, reps, seed):
    """Stationary bootstrap distribution of annualized Sharpe (365 days).

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::block_bootstrap_sharpe.

    mean_block is PENDING in the monograph (rule: >= mean holding period,
    frozen before looking), so the caller must pass it.
    """
    stationary_bootstrap = _arch("StationaryBootstrap")
    block = math.ceil(_require(mean_block, "bootstrap_block_rule"))  # keep block >= frozen mean
    values = np.asarray(pnl, dtype=float).ravel()
    bs = stationary_bootstrap(block, values, seed=seed)
    dist = np.array(
        [_annualized_sharpe(sample[0][0]) for sample in bs.bootstrap(reps)],
        dtype=float,
    )
    low, high = np.quantile(dist, [0.05, 0.95])
    return {
        "sharpe_dist": dist,
        "p_le_zero": float(np.mean(dist <= 0.0)),
        "interval_90": (float(low), float(high)),
    }


def _hac_ols(y, x, maxlags):
    """OLS of y on [1, x] with a Newey-West HAC covariance (Bartlett kernel).

    Replaces `statsmodels.OLS.fit(cov_type="HAC", cov_kwds={"maxlags": lags})` from the
    source function; same kernel, same ``n/(n-k)`` small-sample correction.
    """
    design = np.column_stack([np.ones_like(x), x])
    nobs, k = design.shape
    if nobs <= k:
        raise ValueError("not enough observations for a HAC regression")
    xtx_inv = np.linalg.inv(design.T @ design)
    params = xtx_inv @ design.T @ y
    resid = y - design @ params
    xu = design * resid[:, None]
    meat = xu.T @ xu
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        gamma = xu[lag:].T @ xu[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    cov *= nobs / (nobs - k)
    return params, cov


def beta_adjusted_alpha(strategy_pnl, benchmark_pnl, hac_maxlags):
    """HAC OLS alpha of strategy on benchmark; one-sided 90% test for alpha > 0.

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/stats.py::beta_adjusted_alpha
    (statsmodels HAC replaced by `_hac_ols`, see the module docstring).

    hac_maxlags is not specified in the monograph, so it is a required argument.
    """
    from scipy.stats import norm

    maxlags = int(_require(hac_maxlags, "hac_maxlags"))
    y = np.asarray(strategy_pnl, dtype=float).ravel()
    x = np.asarray(benchmark_pnl, dtype=float).ravel()
    if y.shape != x.shape:
        raise ValueError("strategy_pnl and benchmark_pnl must have the same length")
    params, cov = _hac_ols(y, x, maxlags)
    alpha = float(params[0])
    beta = float(params[1])
    alpha_se = float(math.sqrt(cov[0, 0]))
    beta_se = float(math.sqrt(cov[1, 1]))
    scale = float(np.linalg.norm(y) / math.sqrt(len(y)))
    t_stat = alpha / alpha_se if alpha_se > 1e-12 * scale else float("nan")
    p_one_sided = float(norm.sf(t_stat))  # HAC p-values use the normal distribution
    return {
        "alpha": alpha,
        "alpha_annualized": alpha * _DAYS_PER_YEAR,
        "alpha_se": alpha_se,
        "beta": beta,
        "beta_se": beta_se,
        "t": t_stat,
        "p_one_sided": p_one_sided,
        "pass_90": bool(alpha > 0.0 and p_one_sided < _ALPHA_LEVEL),
    }


def _as_returns(returns, benchmark):
    """Coerce the wrapper inputs to (T, k) strategy PnL and a (T,) benchmark PnL."""
    values = np.asarray(returns, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("returns must be one- or two-dimensional")
    if benchmark is None:
        bench = np.zeros(values.shape[0], dtype=float)
    else:
        bench = np.asarray(benchmark, dtype=float).ravel()
        if bench.shape[0] != values.shape[0]:
            raise ValueError("benchmark and returns must have the same number of rows")
    if not np.isfinite(values).all() or not np.isfinite(bench).all():
        raise ValueError("returns and benchmark must be finite")
    return values, bench


def hansen_spa(returns, *, benchmark=None, reps=1000, block_size=None, seed=None,
               studentize=True, alpha=0.05):
    """Hansen (2005) test of superior predictive ability, via `arch.bootstrap.SPA`.

    Thin wrapper: `returns` is T x k per-period PnL (higher is better) and the test asks
    whether any column beats `benchmark` (default: zero, i.e. "is any strategy positive").
    arch's SPA works on losses, so the sign is flipped here and nowhere else. The headline
    p-value is Hansen's consistent estimator; all three (`lower`, `consistent`, `upper`)
    are returned.
    """
    spa_class = _arch("SPA")
    values, bench = _as_returns(returns, benchmark)
    test = spa_class(
        -bench[:, None], -values,
        block_size=block_size, reps=reps, studentize=studentize, seed=seed,
    )
    test.compute()
    pvalues = test.pvalues
    better = [int(index) for index in test.better_models(alpha)]
    return {
        "p_lower": float(pvalues["lower"]),
        "p_consistent": float(pvalues["consistent"]),
        "p_upper": float(pvalues["upper"]),
        "reject": bool(pvalues["consistent"] <= alpha),
        "better_models": better,
        "n_models": int(values.shape[1]),
        "n_obs": int(values.shape[0]),
        "block_size": int(test.block_size),
        "reps": int(reps),
        "alpha": float(alpha),
    }


def romano_wolf_stepm(returns, *, benchmark=None, size=0.05, reps=1000, block_size=None,
                      seed=None, studentize=True):
    """Romano & Wolf (2005) stepwise multiple comparison, via `arch.bootstrap.StepM`.

    Thin wrapper over `StepM`: same sign convention as `hansen_spa`, FWER controlled at
    `size`. Returns the indices of the models that are superior to the benchmark after the
    stepwise elimination (arch's `superior_models`), not per-model adjusted p-values.
    """
    stepm_class = _arch("StepM")
    values, bench = _as_returns(returns, benchmark)
    test = stepm_class(
        -bench[:, None], -values,
        size=size, block_size=block_size, reps=reps, studentize=studentize, seed=seed,
    )
    test.compute()
    superior = [int(index) for index in test.superior_models]
    return {
        "superior_models": superior,
        "n_superior": len(superior),
        "n_models": int(values.shape[1]),
        "n_obs": int(values.shape[0]),
        "block_size": int(test.block_size),
        "reps": int(reps),
        "size": float(size),
    }
