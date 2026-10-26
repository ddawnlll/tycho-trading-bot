"""Expert ensemble -> per-column signals for the shared-account engines (tycho and Nautilus)."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .engine import Costs, Signals
from .experts import CATALOG, Expert
from .marketstate import MarketState, build_state
from .portfolio import MAJORS


def configured_experts(spec=None) -> list[Expert]:
    out = []
    for name, params, req in (spec or C.EXPERTS):
        base = CATALOG[name]
        out.append(Expert(name, base.rule, base.requires, dict(params), dict(req)))
    return out


def build_columns(data: dict[str, pd.DataFrame], experts: list[Expert], ms: MarketState | None = None,
                  slippage: bool = True):
    """Columns are '<expert>:<symbol>' in priority order (experts first, then universe order)."""
    ms = ms or build_state(data, C.REGIME_N)
    cols, sigs, costs, groups = {}, {}, {}, {}
    for ex in experts:
        for s, df in data.items():
            key = f"{ex.name}:{s}"
            cols[key] = df
            sigs[key] = ex.signals(df, ms.for_symbol(s))
            costs[key] = Costs(slippage=(0.0002 if s in MAJORS else 0.0005) if slippage else 0.0)
            groups[key] = s
    return cols, sigs, costs, groups


def window(cols: dict[str, pd.DataFrame], sigs: dict[str, Signals], start, end):
    d, s = {}, {}
    for k, df in cols.items():
        m = (df.index >= start) & (df.index <= end)
        if m.sum() > 50:
            d[k] = df[m]
            s[k] = type(sigs[k])(*(getattr(sigs[k], f)[m] for f in sigs[k].__dataclass_fields__))
    return d, s
