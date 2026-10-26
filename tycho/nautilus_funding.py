"""NautilusTrader backtest with perpetual funding simulated inside the venue.

Funding is charged on the Nautilus account itself (SimulationModule -> exchange.adjust_account), so
position sizing sees the drag exactly as in a real account. An overlay subtracted afterwards is
optimistic: Nautilus would keep sizing on pre-funding equity (measured: +11% on a long-hold strategy).

Costs: taker fee per instrument = exchange fee + modelled slippage (bar fills have no spread).
Reuses tycho.nautilus_bt (frozen file) for instruments, bars and the EnsembleStrategy.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.config import LoggingConfig, SimulationModuleConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.objects import Money

from . import nautilus_bt as NB

BAR_NS = NB.BAR_NS


class FundingModule(SimulationModule):
    """Charges qty * bar open * summed funding rate of the bar, at the bar's close timestamp, for
    every open position (longs pay positive funding). Same convention as tycho.engine_multi."""

    def __init__(self, schedule: dict[str, dict[int, tuple[float, float]]]):
        super().__init__(SimulationModuleConfig())
        self.schedule = schedule        # instrument id str -> {bar close ns: (open, rate)}
        self.total = 0.0
        self._last = -1

    def process(self, ts_now: int):
        if ts_now == self._last:
            return
        self._last = ts_now
        pay = 0.0
        for pos in self.exchange.cache.positions_open():
            row = self.schedule.get(str(pos.instrument_id), {}).get(ts_now)
            if row is not None:
                pay += float(pos.signed_qty) * row[0] * row[1]
        if pay != 0.0:
            self.total += pay
            self.exchange.adjust_account(Money(-pay, USDT))

    def log_diagnostics(self, logger):
        pass

    def reset(self):
        self.total = 0.0
        self._last = -1


def run(data: dict[str, pd.DataFrame], cols, sigs, start, end, filters: dict[str, dict], taker_fee: dict[str, float],
        equity0: float = 10_000.0, **cfg) -> dict:
    """NB.run plus in-venue funding and per-instrument cost; data must already be instrument-rounded."""
    inst = {s: NB.make_instrument(s, filters[s], taker_fee=taker_fee[s]) for s in data}
    schedule = {}
    for s, df in data.items():
        w = df[(df.index >= start) & (df.index <= end) & (df["funding"] != 0)]
        schedule[str(inst[s].id)] = {int(t.value) + BAR_NS: (float(o), float(f))
                                     for t, o, f in zip(w.index, w["open"], w["funding"])}
    funding = FundingModule(schedule)
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), run_analysis=False))
    engine.add_venue(NB.VENUE, OmsType.NETTING, AccountType.MARGIN, [Money(equity0, USDT)], base_currency=USDT,
                     default_leverage=Decimal(10), bar_execution=True, modules=[funding])
    bar_types, expected = {}, {}
    for s, df in data.items():
        engine.add_instrument(inst[s])
        bt = BarType.from_str(f"{inst[s].id}-4-HOUR-LAST-EXTERNAL")
        bar_types[s] = bt
        w = df[(df.index >= start) & (df.index <= end)]
        if w.empty:
            continue
        engine.add_data(NB.make_bars(bt, inst[s], w))
        for t in w.index.as_unit("ns").asi8:
            expected[t] = expected.get(t, 0) + 1
    strat = NB.EnsembleStrategy(NB.EnsembleConfig(**cfg), NB.plan_frames(sigs, cols), inst, bar_types, expected)
    engine.add_strategy(strat)
    engine.run()
    out = dict(positions=engine.trader.generate_positions_report(), fills=engine.trader.generate_order_fills_report(),
               equity=pd.Series({pd.Timestamp(t, tz="UTC"): v for t, v in strat.equity_log}).sort_index(),
               funding_paid=funding.total)
    engine.dispose()
    return out
