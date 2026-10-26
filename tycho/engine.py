"""Bar-by-bar single-asset backtester for perpetual futures.

Timing contract (no lookahead):
  * Strategy arrays at index i may only use data up to and including bar i's close.
  * Entries/exits decided at close of bar i fill at OPEN of bar i+1 (+slippage).
  * Protective stops are resting orders checked intrabar from bar i+1 on; a gap
    through the stop fills at the bar open (worse price), else at the stop price.
  * Funding: the bar's summed funding rate is charged on notional held during it
    (longs pay positive funding, shorts receive it).
Sizing: risk-per-trade fraction of equity / stop distance, capped by max leverage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Costs:
    fee: float = 0.0005       # Binance USDT-M VIP0 taker 0.05% per side
    slippage: float = 0.0002  # 2 bps per side on BTC; raise for alts


@dataclass
class Signals:
    long_entry: np.ndarray    # bool[i]: enter long at open i+1
    short_entry: np.ndarray
    long_stop: np.ndarray     # initial stop price if entered at i+1
    short_stop: np.ndarray
    long_trail: np.ndarray    # candidate trailing stop computed at close i (ratchets)
    short_trail: np.ndarray
    long_exit: np.ndarray     # bool[i]: close-based exit at open i+1
    short_exit: np.ndarray


def run(df: pd.DataFrame, sig: Signals, risk: float = 0.01, max_lev: float = 3.0,
        costs: Costs = Costs(), equity0: float = 10_000.0) -> dict:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    fund = df["funding"].to_numpy(float) if "funding" in df else np.zeros(len(df))
    n = len(df)
    eq = np.empty(n)
    cash = equity0
    pos = 0.0          # signed qty
    entry = stop = r_unit = 0.0
    entry_i = -1
    trades = []

    def close_pos(i, px, reason):
        nonlocal cash, pos
        side = np.sign(pos)
        fill = px * (1 - side * costs.slippage)
        cash += pos * (fill - entry) - abs(pos) * fill * costs.fee
        trades.append(dict(entry_time=df.index[entry_i], exit_time=df.index[i], side=int(side),
                           entry=entry, exit=fill, qty=abs(pos),
                           r=(side * (fill - entry) * abs(pos) - abs(pos) * (entry + fill) * costs.fee) / r_unit,
                           reason=reason, bars=i - entry_i))
        pos = 0.0

    for i in range(n):
        # 1) orders decided at close i-1 fill at open i
        if i > 0:
            if pos > 0 and sig.long_exit[i - 1]:
                close_pos(i, o[i], "signal")
            elif pos < 0 and sig.short_exit[i - 1]:
                close_pos(i, o[i], "signal")
            if pos == 0:
                for side, ent, stp in ((1, sig.long_entry, sig.long_stop), (-1, sig.short_entry, sig.short_stop)):
                    if ent[i - 1] and np.isfinite(stp[i - 1]):
                        px = o[i] * (1 + side * costs.slippage)
                        dist = side * (px - stp[i - 1])
                        if dist <= 0:
                            continue  # gapped past the stop already; skip
                        equity = cash
                        qty = min(equity * risk / dist, equity * max_lev / px)
                        cash -= qty * px * costs.fee
                        pos, entry, stop, entry_i = side * qty, px, stp[i - 1], i
                        r_unit = qty * dist
                        break
        # 2) funding on position held through bar i
        if pos != 0:
            cash -= pos * o[i] * fund[i]
        # 3) intrabar stop
        if pos > 0 and l[i] <= stop:
            close_pos(i, min(o[i], stop), "stop")
        elif pos < 0 and h[i] >= stop:
            close_pos(i, max(o[i], stop), "stop")
        # 4) mark to market, ratchet trailing stop at close
        eq[i] = cash + (pos * (c[i] - entry) if pos else 0.0)
        if pos > 0 and np.isfinite(sig.long_trail[i]):
            stop = max(stop, sig.long_trail[i])
        elif pos < 0 and np.isfinite(sig.short_trail[i]):
            stop = min(stop, sig.short_trail[i])
        if eq[i] <= 0:  # liquidated: book the loss and stop trading
            if pos != 0:
                close_pos(i, c[i], "liquidated")
            eq[i:] = 0.0
            break
    if pos != 0:
        close_pos(n - 1, c[-1], "eod")
        eq[-1] = cash
    tr = pd.DataFrame(trades)
    return dict(equity=pd.Series(eq, index=df.index, name="equity"), trades=tr)


def metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    daily = equity.resample("1D").last().ffill()
    rets = daily.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    dd = (equity / equity.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0.0
    m = dict(start=str(equity.index[0].date()), end=str(equity.index[-1].date()),
             total_return=float(total), cagr=float(cagr), max_dd=float(dd),
             sharpe=float(sharpe), mar=float(cagr / abs(dd)) if dd < 0 else 0.0,
             n_trades=int(len(trades)))
    if len(trades):
        r = trades["r"]
        wins, losses = r[r > 0], r[r <= 0]
        m.update(win_rate=float((r > 0).mean()), avg_r=float(r.mean()),
                 avg_win_r=float(wins.mean()) if len(wins) else 0.0,
                 avg_loss_r=float(losses.mean()) if len(losses) else 0.0,
                 max_r=float(r.max()),
                 profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
                 long_share=float((trades["side"] > 0).mean()))
    return m


def cfg_dict(obj) -> dict:
    return asdict(obj)
