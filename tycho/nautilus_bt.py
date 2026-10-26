"""NautilusTrader backtest of the expert ensemble (independent execution/accounting engine).

Signals come from the same causal expert functions (tycho.ensemble.build_columns); NautilusTrader
does order handling, fills (bar OHLC), stop triggers, fees, margin and position accounting.

Timing: Nautilus processes each bar before Strategy.on_bar, so a market order sent in on_bar fills
at that bar's CLOSE (tycho fills at the next OPEN; median gap 0, p99 < 0.1% on these perps).
Bars are stamped at their close time. Decisions for a timestamp are made once, after every
symbol's bar for that timestamp has arrived. Funding is not simulated by Nautilus.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, CurrencyType, OmsType, OrderSide, TriggerType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from . import config as C
from .engine import Signals

VENUE = Venue("BINANCE")
INFO_PATH = Path(__file__).resolve().parent.parent / "data" / "exchange_info.json"
BAR_NS = int(pd.Timedelta(C.TIMEFRAME).value)


def _decimals(step: str) -> int:
    d = Decimal(step).normalize()
    return max(-d.as_tuple().exponent, 0)


def exchange_filters() -> dict[str, dict]:
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    out = {}
    for s in info["symbols"]:
        f = {x["filterType"]: x for x in s["filters"]}
        out[s["symbol"]] = dict(tick=f["PRICE_FILTER"]["tickSize"], step=f["LOT_SIZE"]["stepSize"])
    return out


def make_instrument(symbol: str, filt: dict, taker_fee: float = 0.0005) -> CryptoPerpetual:
    base = symbol[:-4]
    pp, sp = _decimals(filt["tick"]), _decimals(filt["step"])
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol}-PERP"), VENUE), raw_symbol=Symbol(symbol),
        base_currency=Currency(base, 8, 0, base, CurrencyType.CRYPTO), quote_currency=USDT,
        settlement_currency=USDT, is_inverse=False,
        price_precision=pp, price_increment=Price(Decimal(filt["tick"]), pp),
        size_precision=sp, size_increment=Quantity(Decimal(filt["step"]), sp),
        max_quantity=None, min_quantity=Quantity(Decimal(filt["step"]), sp), max_notional=None, min_notional=None,
        max_price=None, min_price=None, margin_init=Decimal("0.05"), margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.0002"), taker_fee=Decimal(str(taker_fee)), ts_event=0, ts_init=0)


def round_to_instrument(df: pd.DataFrame, inst: CryptoPerpetual) -> pd.DataFrame:
    """Nautilus requires bar prices at instrument precision; both engines get this same data."""
    out = df.copy()
    p = inst.price_precision
    for k in ("open", "high", "low", "close"):
        out[k] = out[k].round(p)
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


class EnsembleConfig(StrategyConfig, frozen=True):
    risk: float = C.RISK_DEFAULT
    max_positions: int = C.MAX_POSITIONS
    max_gross: float = C.MAX_GROSS
    max_lev_pos: float = C.MAX_LEV_POS


class EnsembleStrategy(Strategy):
    """Executes precomputed per-column signals. Columns '<expert>:<symbol>' in priority order;
    one expert per symbol (Binance one-way netting). Long-only experts."""

    def __init__(self, config: EnsembleConfig, plan: dict[str, pd.DataFrame], instruments: dict[str, CryptoPerpetual],
                 bar_types: dict[str, BarType], expected: dict[int, int]):
        super().__init__(config)
        self.plan = plan                      # col -> DataFrame[long_entry, long_stop, long_trail, long_exit] by bar OPEN ns
        self.inst = instruments               # symbol -> instrument
        self.bar_types = bar_types
        self.expected = expected              # bar open ns -> number of symbols with a bar
        self.seen: dict[int, int] = {}
        self.close: dict[str, float] = {}
        self.held: dict[str, dict] = {}       # symbol -> {col, stop, stop_order, pending}
        self.equity_log: list[tuple[int, float]] = []
        self._cancelling: set = set()        # client ids we cancel on purpose
        self.cols = list(plan)

    def on_start(self):
        for bt in self.bar_types.values():
            self.subscribe_bars(bt)

    # ------------------------------------------------------------------ helpers
    def _equity(self) -> float:
        acct = self.portfolio.account(VENUE)
        bal = float(acct.balance_total(USDT).as_double())
        upnl = self.portfolio.unrealized_pnls(VENUE)
        return bal + sum(float(m.as_double()) for m in upnl.values())

    def _gross(self) -> float:
        return sum(abs(float(self.portfolio.net_position(self.inst[s].id))) * self.close[s] for s in self.held)

    # ------------------------------------------------------------------ events
    def on_bar(self, bar: Bar):
        sym = bar.bar_type.instrument_id.symbol.value.replace("-PERP", "")
        self.close[sym] = float(bar.close)
        t_open = bar.ts_event - BAR_NS
        self.seen[t_open] = self.seen.get(t_open, 0) + 1
        if self.seen[t_open] == self.expected.get(t_open, -1):
            self._decide(t_open)

    def _row(self, col, t_open):
        df = self.plan[col]
        return df.loc[t_open] if t_open in df.index else None

    def _decide(self, t_open: int):
        self._sync_held()
        # 1) ratchet trailing stops (tycho: at close i, effective from bar i+1)
        for s, h in list(self.held.items()):
            r = self._row(h["col"], t_open)
            if r is None or h["stop_order"] is None:
                continue
            tr = r["long_trail"]
            if np.isfinite(tr) and tr > h["stop"]:
                newp = self.inst[s].make_price(tr)
                if float(newp) < self.close[s]:
                    self._replace_stop(s, float(newp))
                else:  # trail already through the market: exit now (tycho would stop next bar)
                    self._exit(s, "trail_through")
        # 2) signal exits
        for s, h in list(self.held.items()):
            r = self._row(h["col"], t_open)
            if r is not None and bool(r["long_exit"]):
                self._exit(s, "signal")
        # 3) entries in priority order
        eq = self._equity()
        for col in self.cols:
            if len(self.held) >= self.config.max_positions:
                break
            s = col.split(":")[1]
            if s in self.held or s not in self.close:
                continue
            r = self._row(col, t_open)
            if r is None or not bool(r["long_entry"]) or not np.isfinite(r["long_stop"]):
                continue
            px, stop = self.close[s], float(r["long_stop"])
            dist = px - stop
            if dist <= 0:
                continue
            room = max(eq * self.config.max_gross - self._gross(), 0.0)
            q = min(eq * self.config.risk / dist, eq * self.config.max_lev_pos / px, room / px)
            inst = self.inst[s]
            step = float(inst.size_increment)
            q = np.floor(q / step) * step
            if q < step:
                continue
            order = self.order_factory.market(inst.id, OrderSide.BUY, inst.make_qty(q))
            self.held[s] = dict(col=col, stop=float(inst.make_price(stop)), stop_order=None, entry_order=order.client_order_id)
            self.submit_order(order)
        self.equity_log.append((t_open + BAR_NS, eq))

    # ------------------------------------------------------------------ order management
    def _sync_held(self):
        """Drop coins whose protected position is gone (e.g. a stop fill this strategy didn't see)."""
        for s, h in list(self.held.items()):
            if h["stop_order"] is not None and float(self.portfolio.net_position(self.inst[s].id)) == 0:
                self._cancel(h["stop_order"])
                self.held.pop(s)

    def _cancel(self, client_order_id):
        o = self.cache.order(client_order_id) if client_order_id is not None else None
        if o is not None and o.is_open:
            self._cancelling.add(client_order_id)
            self.cancel_order(o)

    def _submit_stop(self, s: str, stop: float):
        inst = self.inst[s]
        qty = float(self.portfolio.net_position(inst.id))
        if qty <= 0:
            return None
        o = self.order_factory.stop_market(inst.id, OrderSide.SELL, inst.make_qty(qty), inst.make_price(stop),
                                           trigger_type=TriggerType.LAST_PRICE, reduce_only=True)
        self.submit_order(o)
        return o.client_order_id

    def _replace_stop(self, s: str, stop: float):
        """New stop first, then cancel the old one (Binance rejects modify on STOP_MARKET)."""
        h = self.held[s]
        old = h["stop_order"]
        new = self._submit_stop(s, stop)
        if new is None:
            return
        h["stop_order"], h["stop"] = new, stop
        self._cancel(old)

    def _exit(self, s: str, reason: str):
        h = self.held.get(s)
        if h is None:
            return
        self._cancel(h["stop_order"])
        qty = self.portfolio.net_position(self.inst[s].id)
        if qty > 0:
            self.submit_order(self.order_factory.market(self.inst[s].id, OrderSide.SELL,
                                                        self.inst[s].make_qty(float(qty)), reduce_only=True))
        self.held.pop(s, None)

    def _sym(self, event) -> str:
        return event.instrument_id.symbol.value.replace("-PERP", "")

    def on_order_filled(self, event):
        s = self._sym(event)
        h = self.held.get(s)
        if h is None:
            return
        if event.client_order_id == h.get("entry_order"):
            entry = self.cache.order(event.client_order_id)
            if h["stop_order"] is None and entry is not None and entry.is_closed:  # protect the FULL fill
                h["stop_order"] = self._submit_stop(s, h["stop"])
        elif event.client_order_id == h.get("stop_order"):
            self.held.pop(s, None)

    def _order_failed(self, event, what: str):
        s = self._sym(event)
        h = self.held.get(s)
        cid = event.client_order_id
        if cid in self._cancelling:
            self._cancelling.discard(cid)
            return
        if h is None:
            return
        if cid == h.get("entry_order"):
            if float(self.portfolio.net_position(self.inst[s].id)) == 0:
                self.held.pop(s, None)        # entry never happened: free the slot
            elif h["stop_order"] is None:
                h["stop_order"] = self._submit_stop(s, h["stop"])  # partially filled then failed
        elif cid == h.get("stop_order"):
            self.log.error(f"stop {what} for {s}; flattening")
            h["stop_order"] = None
            self._exit(s, f"stop_{what}")

    def on_order_rejected(self, event):
        self._order_failed(event, "rejected")

    def on_order_denied(self, event):
        self._order_failed(event, "denied")

    def on_order_canceled(self, event):
        self._order_failed(event, "canceled")

    def on_order_expired(self, event):
        self._order_failed(event, "expired")


def make_bars(bt: BarType, inst: CryptoPerpetual, df: pd.DataFrame) -> list[Bar]:
    """Bars stamped at CLOSE time (bar open + timeframe). Built directly: pandas copy-on-write
    arrays are read-only, which BarDataWrangler rejects."""
    ts = (df.index.as_unit("ns").asi8 + BAR_NS).tolist()
    o, h, l, c, v = (df[k].to_numpy(float).tolist() for k in ("open", "high", "low", "close", "volume"))
    v = [min(x, 1e9) for x in v]  # volume unused by fills; cap below Nautilus QUANTITY_MAX (DOGE > 1.8e10)
    mp, mq = inst.make_price, inst.make_qty
    return [Bar(bt, mp(o[i]), mp(h[i]), mp(l[i]), mp(c[i]), mq(v[i]), ts[i], ts[i]) for i in range(len(ts))]


def plan_frames(sigs: dict[str, Signals], cols: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for k, sg in sigs.items():
        idx = cols[k].index.as_unit("ns").asi8
        out[k] = pd.DataFrame({"long_entry": sg.long_entry, "long_stop": sg.long_stop, "long_trail": sg.long_trail,
                               "long_exit": sg.long_exit}, index=idx)
    return out


def run(data: dict[str, pd.DataFrame], cols, sigs, start, end, equity0: float = 10_000.0, **cfg) -> dict:
    filt = exchange_filters()
    inst = {s: make_instrument(s, filt[s]) for s in data}
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"), run_analysis=False))
    engine.add_venue(VENUE, OmsType.NETTING, AccountType.MARGIN, [Money(equity0, USDT)], base_currency=USDT,
                     default_leverage=Decimal(10), bar_execution=True)
    bar_types, expected = {}, {}
    for s, df in data.items():
        engine.add_instrument(inst[s])
        bt = BarType.from_str(f"{inst[s].id}-4-HOUR-LAST-EXTERNAL")
        bar_types[s] = bt
        w = df[(df.index >= start) & (df.index <= end)]
        if w.empty:
            continue
        engine.add_data(make_bars(bt, inst[s], w))
        for t in w.index.as_unit("ns").asi8:
            expected[t] = expected.get(t, 0) + 1
    plan = plan_frames(sigs, cols)
    strat = EnsembleStrategy(EnsembleConfig(**cfg), plan, inst, bar_types, expected)
    engine.add_strategy(strat)
    engine.run()
    pos = engine.trader.generate_positions_report()
    fills = engine.trader.generate_order_fills_report()
    acct = engine.trader.generate_account_report(VENUE)
    orders = engine.trader.generate_orders_report()
    open_orders = len(engine.cache.orders_open())
    held_end = dict(strat.held)
    eq = pd.Series({pd.Timestamp(t, tz="UTC"): v for t, v in strat.equity_log}).sort_index()
    engine.dispose()
    return dict(equity=eq, positions=pos, fills=fills, account=acct, orders=orders, open_orders=open_orders,
                held_end=held_end)
