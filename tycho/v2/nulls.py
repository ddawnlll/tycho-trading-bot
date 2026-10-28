"""Null strategies for the K1 negative control (v2).

Copied from `ddawnlll/tycho` at commit `0a309c4`, file `src/tycho/nulls.py`:
`eligible_days`, `blocks`, `apply_gross_cap`, `scramble` (D-40a) and `random_entry`
(D-40b), plus the `_by_asset` / `_check` / `_validate` helpers and the module constants.

Not copied: `no_trade`, `equal_risk_buy_and_hold`, `trend_following_btc`, `run_baselines`
and `run_nulls`. Those are wrappers that execute the E-001 ledger (`e001.build_experiment`,
`e001._simulate`) and append trials; the E-001 pipeline does not exist in v2 yet, and
G0-T1 only asks for the null *generators* ("random-entry and scramble nulls"). The
generators are self-contained: they take a realized weight panel plus an eligibility
panel and are pure functions of them.

Every null is built from the strategy's realized target-weight panel ``w[t, asset]``.
Eligible days for an asset are the days it is in the point-in-time universe and listed. A
block is a maximal run of consecutive eligible days with the same sign of ``w`` (+, -, 0);
ineligible days are skipped by working on the compressed eligible-day index.
"""

import numpy as np
import pandas as pd

BTC = "BTCUSDT"
GROSS_CAP = 2.0


def eligible_days(base) -> pd.DataFrame:
    """PIT universe membership and a listed bar (the D-40 eligible days).

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::eligible_days.
    """
    return base["member"] & base["panels"]["close"].notna()


def blocks(w, eligible) -> list[dict]:
    """Per-asset maximal same-sign runs on the compressed eligible-day index.

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::blocks.
    """
    _validate(w, eligible)
    out = []
    for asset in w.columns:
        positions = np.flatnonzero(eligible[asset].to_numpy(dtype=bool))
        if positions.size == 0:
            continue
        values = w[asset].to_numpy(dtype=float)[positions]
        signs = np.sign(values)
        starts = np.r_[0, np.flatnonzero(np.diff(signs) != 0) + 1]
        ends = np.r_[starts[1:], len(signs)]
        for start, end in zip(starts, ends):
            out.append({
                "asset": asset,
                "start": w.index[positions[start]],
                "end": w.index[positions[end - 1]],
                "sign": int(signs[start]),
                "length": int(end - start),
                "_start": int(start),
                "_end": int(end),
            })
    return out


def apply_gross_cap(weights, cap=GROSS_CAP):
    """Proportional date-wise gross-cap shrink; returns (panel, binding-day count).

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::apply_gross_cap.
    """
    out = weights.astype(float).copy()
    gross = out.abs().sum(axis=1)
    binding = gross > cap
    if binding.any():
        out.loc[binding] = out.loc[binding].mul(cap / gross[binding], axis=0)
    return out, int(binding.sum())


def scramble(w, eligible, *, reps, seed):
    """Run-based direction scramble (D-40a).

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::scramble.

    Preserved exactly: ``|w|`` for every (t, asset); date-wise gross; the original
    block boundaries (reported as metadata).
    Not preserved: realized holding period (merges allowed), net exposure path,
    cross-sectional sign correlation, turnover, costs (recomputed).

    Each non-flat block independently draws s in {+1, -1} with p=0.5; flat blocks
    stay flat. Neighbouring blocks may take the same sign and merge.
    """
    _check(reps, w, eligible)
    per_asset = _by_asset(blocks(w, eligible))
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        frame = pd.DataFrame(0.0, index=w.index, columns=w.columns)
        metadata = []
        for asset in w.columns:
            positions = np.flatnonzero(eligible[asset].to_numpy(dtype=bool))
            if positions.size == 0:
                continue
            values = w[asset].to_numpy(dtype=float)
            null = np.zeros(len(w))
            for block in per_asset.get(asset, []):
                sign = 0
                if block["sign"] != 0:
                    sign = 1 if rng.integers(0, 2) else -1
                    segment = positions[block["_start"]:block["_end"]]
                    null[segment] = sign * np.abs(values[segment])
                metadata.append({key: value for key, value in {
                    **block, "scramble_sign": sign,
                }.items() if not key.startswith("_")})
            frame[asset] = null
        yield {"weights": frame, "blocks": metadata}


def random_entry(w, eligible, *, reps, seed, gross_cap=GROSS_CAP):
    """Run-preserving random entry (D-40b).

    Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::random_entry.

    Preserved exactly: per-asset run-length multiset; per-asset flat-length
    multiset; pre-cap per-asset total ``|w|``; eligibility (no position on an
    ineligible day).
    Not preserved: date-wise cross-sectional gross, net exposure path,
    cross-sectional correlation, turnover, costs (recomputed).

    Runs and flats are separated and each list is shuffled with a seeded
    Fisher-Yates permutation; the shuffled blocks are rebuilt on the compressed
    eligible-day index in the original run/flat alternation, each run keeping its
    own ``|w|`` profile and drawing a sign. Where two runs are adjacent (no flat
    between them, a sign change in the original), the second sign is forced
    opposite so that the preserved run-length multiset stays observable. The
    result is mapped back to calendar days and the strategy's gross cap (2.0) is
    applied with a proportional shrink; the volatility cap is not reapplied.
    ``metadata`` reports ``cap_bind_days`` for the repetition.
    """
    _check(reps, w, eligible)
    per_asset = _by_asset(blocks(w, eligible))
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        frame = pd.DataFrame(0.0, index=w.index, columns=w.columns)
        pre_cap_total = {}
        for asset in w.columns:
            positions = np.flatnonzero(eligible[asset].to_numpy(dtype=bool))
            if positions.size == 0:
                continue
            values = w[asset].to_numpy(dtype=float)
            asset_blocks = per_asset.get(asset, [])
            runs = [b for b in asset_blocks if b["sign"] != 0]
            flats = [b for b in asset_blocks if b["sign"] == 0]
            runs = [runs[i] for i in rng.permutation(len(runs))]
            flats = [flats[i] for i in rng.permutation(len(flats))]
            null = np.zeros(len(positions))
            cursor = run_i = flat_i = 0
            previous_was_run = False
            previous_sign = 0
            for block in asset_blocks:
                if block["sign"] != 0:
                    source = runs[run_i]
                    run_i += 1
                    sign = 1 if rng.integers(0, 2) else -1
                    # Adjacent runs must stay opposite or their lengths would merge
                    # and the run-length multiset would no longer be observable.
                    if previous_was_run and sign == previous_sign:
                        sign = -sign
                    segment = positions[source["_start"]:source["_end"]]
                    values_segment = np.abs(values[segment])
                    null[cursor:cursor + source["length"]] = sign * values_segment
                    previous_was_run, previous_sign = True, sign
                else:
                    source = flats[flat_i]
                    flat_i += 1
                    null[cursor:cursor + source["length"]] = 0.0
                    previous_was_run, previous_sign = False, 0
                cursor += source["length"]
            if cursor != len(positions):
                raise AssertionError("random entry lost compressed eligible days")
            frame.loc[frame.index[positions], asset] = null
            pre_cap_total[asset] = float(np.abs(null).sum())
        capped, binds = apply_gross_cap(frame, gross_cap)
        yield {
            "weights": capped,
            "pre_cap_total_abs": pre_cap_total,
            "cap_bind_days": binds,
        }


def _by_asset(items):
    """Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::_by_asset."""
    out = {}
    for item in items:
        out.setdefault(item["asset"], []).append(item)
    return out


def _check(reps, w, eligible):
    """Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::_check."""
    if type(reps) is not int or reps < 1:
        raise ValueError("reps must be a positive integer")
    _validate(w, eligible)


def _validate(w, eligible):
    """Source: ddawnlll/tycho @ 0a309c4, src/tycho/nulls.py::_validate."""
    if not isinstance(w, pd.DataFrame) or not isinstance(eligible, pd.DataFrame):
        raise ValueError("w and eligible must be DataFrames")
    if not w.index.equals(eligible.index) or not w.columns.equals(eligible.columns):
        raise ValueError("w and eligible must share index and columns")
    mask = eligible.to_numpy(dtype=bool)
    values = w.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("w must be finite")
    if np.any(values[~mask] != 0.0):
        raise ValueError("realized weights must be flat outside eligible days")
