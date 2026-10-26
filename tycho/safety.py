"""Small fail-closed boundaries adapted from ddawnlll/tycho runtime contracts."""
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hash(path):
    # Git may convert LF/CRLF across Windows and CI; semantic source identity must not.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def expert_manifest(expert):
    return canonical_hash(dict(name=expert.name, params=expert.params, requires=expert.req_params,
                               rule=inspect.getsource(expert.rule), gate=inspect.getsource(expert.requires),
                               dependencies={name: source_hash(Path(__file__).with_name(name))
                                             for name in ("experts.py", "strategies.py", "marketstate.py", "ensemble.py",
                                                          "config.py", "nautilus_bt.py", "nautilus_live.py", "safety.py")}))


def paper_only(mode, risk):
    if mode not in {"paper", "sandbox"}:
        raise ValueError("Only Nautilus paper/sandbox execution is enabled; exchange execution is disabled")
    if not math.isfinite(risk) or not 0 < risk <= 0.02:
        raise ValueError("risk must be finite and in (0, 0.02]")
    if len(C.EXPERTS) != 2 or {n for n, _, _ in C.EXPERTS} != {"trend_long", "xs_momentum"}:
        raise ValueError("Paper observation is authorized only for trend_long and xs_momentum")


def validate_bars(bars, expected_open):
    """Complete synchronized universe, UTC, contiguous closed bars, valid OHLCV.

    Reject the entire decision rather than silently changing breadth/momentum ranks.
    """
    if set(bars) != set(C.UNIVERSE):
        raise ValueError("incomplete universe; abstain")
    for symbol, df in bars.items():
        needed = C.REGIME_WARMUP_BARS if symbol == "BTCUSDT" else C.WARMUP_BARS
        idx = df.index
        if len(df) < needed or not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
            raise ValueError(f"{symbol}: insufficient warmup or missing UTC clock")
        if not idx.is_unique or not idx.is_monotonic_increasing or idx[-1] != expected_open:
            raise ValueError(f"{symbol}: duplicate, future or stale bars")
        if not (idx[1:] - idx[:-1] == pd.Timedelta(C.TIMEFRAME)).all():
            raise ValueError(f"{symbol}: bar gap")
        values = df[["open", "high", "low", "close", "volume"]].to_numpy(float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError(f"{symbol}: nonfinite/nonpositive OHLCV")
        if (df.high < df[["open", "close", "low"]].max(axis=1)).any() or (
                df.low > df[["open", "close", "high"]].min(axis=1)).any():
            raise ValueError(f"{symbol}: inconsistent OHLC")
