"""Shared-equity multi-asset backtester (one account, like the live bot).

Same timing contract as engine.run (decide at close i, fill at open i+1, intrabar stops with gap
fills, per-bar funding). Adds portfolio constraints:
  * max_positions: concurrent open positions
  * max_gross: total notional / equity cap
  * when more entries than free slots, take the highest `rank` (e.g. relative strength) first.
With one symbol and loose caps it must reproduce engine.run exactly (see tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Costs, Signals


def align(data: dict[str, pd.DataFrame], sigs: dict[str, Signals]):
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    syms = list(data)
    T, N = len(idx), len(syms)
    A = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "funding")}
    S = {f: np.full((T, N), np.nan) for f in Signals.__dataclass_fields__}
    for j, s in enumerate(syms):
        pos = idx.get_indexer(data[s].index)
        for k in A:
            A[k][pos, j] = data[s][k].to_numpy(float)
        for f in S:
            S[f][pos, j] = getattr(sigs[s], f).astype(float)
    A["funding"] = np.nan_to_num(A["funding"])
    for f in ("long_entry", "short_entry", "long_exit", "short_exit"):
        S[f] = np.nan_to_num(S[f]).astype(bool)
    return idx, syms, A, S


def run_multi(data: dict[str, pd.DataFrame], sigs: dict[str, Signals], rank: dict[str, np.ndarray] | None = None,
              risk: float = 0.005, max_lev_pos: float = 1.0, max_gross: float = 3.0, max_positions: int = 6,
              costs: dict[str, Costs] | Costs = Costs(), equity0: float = 10_000.0) -> dict:
    idx, syms, A, S = align(data, sigs)
    T, N = len(idx), len(syms)
    o, h, l, c, fund = A["open"], A["high"], A["low"], A["close"], A["funding"]
    R = np.zeros((T, N))
    if rank is not None:
        for j, s in enumerate(syms):
            R[idx.get_indexer(data[s].index), j] = np.nan_to_num(rank[s], nan=-np.inf)
    cst = [costs[s] if isinstance(costs, dict) else costs for s in syms]
    cash = equity0
    qty = np.zeros(N)
    entry = np.zeros(N)
    stop = np.zeros(N)
    r_unit = np.zeros(N)
    entry_i = np.zeros(N, int)
    last = np.full(N, np.nan)  # last known close (for MTM when a symbol has no bar)
    eq = np.empty(T)
    trades = []

    def close(j, i, px, reason):
        nonlocal cash
        side = np.sign(qty[j])
        fill = px * (1 - side * cst[j].slippage)
        q = abs(qty[j])
        cash += qty[j] * (fill - entry[j]) - q * fill * cst[j].fee
        trades.append(dict(symbol=syms[j], entry_time=idx[entry_i[j]], exit_time=idx[i], side=int(side),
                           entry=entry[j], exit=fill, qty=q,
                           r=(side * (fill - entry[j]) * q - q * (entry[j] + fill) * cst[j].fee) / r_unit[j],
                           reason=reason, bars=i - entry_i[j]))
        qty[j] = 0.0

    def equity_at(px):
        m = qty != 0
        return cash + float(np.sum(qty[m] * (px[m] - entry[m])))

    for i in range(T):
        live = ~np.isnan(o[i])
        if i > 0:
            # exits at open
            for j in np.where(live & (qty > 0) & S["long_exit"][i - 1])[0]:
                close(j, i, o[i, j], "signal")
            for j in np.where(live & (qty < 0) & S["short_exit"][i - 1])[0]:
                close(j, i, o[i, j], "signal")
            # entries at open, best rank first
            cand = []
            for j in np.where(live & (qty == 0))[0]:
                if S["long_entry"][i - 1, j] and np.isfinite(S["long_stop"][i - 1, j]):
                    cand.append((R[i - 1, j], j, 1, S["long_stop"][i - 1, j]))
                elif S["short_entry"][i - 1, j] and np.isfinite(S["short_stop"][i - 1, j]):
                    cand.append((R[i - 1, j], j, -1, S["short_stop"][i - 1, j]))
            if cand:
                mark = np.where(np.isnan(o[i]), last, o[i])
                for _, j, side, stp in sorted(cand, key=lambda x: -x[0]):
                    if np.count_nonzero(qty) >= max_positions:
                        break
                    px = o[i, j] * (1 + side * cst[j].slippage)
                    dist = side * (px - stp)
                    if dist <= 0:
                        continue
                    e = equity_at(mark)
                    gross = float(np.sum(np.abs(qty) * np.nan_to_num(mark)))
                    room = max(e * max_gross - gross, 0.0)
                    q = min(e * risk / dist, e * max_lev_pos / px, room / px)
                    if q <= 0:
                        continue
                    cash -= q * px * cst[j].fee
                    qty[j], entry[j], stop[j], entry_i[j] = side * q, px, stp, i
                    r_unit[j] = q * dist
        # funding
        held = live & (qty != 0)
        cash -= float(np.sum(qty[held] * o[i, held] * fund[i, held]))
        # intrabar stops
        for j in np.where(held & (qty > 0) & (l[i] <= stop))[0]:
            close(j, i, min(o[i, j], stop[j]), "stop")
        for j in np.where(held & (qty < 0) & (h[i] >= stop))[0]:
            close(j, i, max(o[i, j], stop[j]), "stop")
        last = np.where(np.isnan(c[i]), last, c[i])
        eq[i] = equity_at(last)
        # trail ratchet
        for j in np.where(live & (qty > 0))[0]:
            if np.isfinite(S["long_trail"][i, j]):
                stop[j] = max(stop[j], S["long_trail"][i, j])
        for j in np.where(live & (qty < 0))[0]:
            if np.isfinite(S["short_trail"][i, j]):
                stop[j] = min(stop[j], S["short_trail"][i, j])
        if eq[i] <= 0:
            for j in np.where(qty != 0)[0]:
                close(j, i, last[j], "liquidated")
            eq[i:] = 0.0
            break
    for j in np.where(qty != 0)[0]:
        close(j, T - 1, last[j], "eod")
    if eq[-1] > 0:
        eq[-1] = cash
    return dict(equity=pd.Series(eq, index=idx, name="equity"), trades=pd.DataFrame(trades))
