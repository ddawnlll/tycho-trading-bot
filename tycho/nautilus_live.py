"""One Nautilus paper node: public Binance data and sandbox execution only.

Network data preparation runs in worker threads; order handling stays on the node
loop. Each process starts a new sandbox account. Persisted ownership is diagnostic,
not a claim that Nautilus sandbox balances/positions survive a process restart.
"""
from __future__ import annotations

import json
import asyncio
from decimal import Decimal

import pandas as pd
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig, LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.currencies import USDT

from . import config as C
from .ensemble import build_columns, configured_experts
from .live import STATE_DIR, fetch_all_bars, last_price, server_now
from .marketstate import build_state
from .nautilus_bt import VENUE, EnsembleConfig, EnsembleStrategy
from .safety import paper_only, validate_bars, expert_manifest
from .reporting import PaperReport
from .data import _get

BAR = pd.Timedelta(C.TIMEFRAME)


def instrument_id(symbol: str) -> InstrumentId:
    return InstrumentId.from_str(f"{symbol}-PERP.BINANCE")


def latest_plan(bars: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Latest-bar signal row per '<expert>:<symbol>' column + MarketState snapshot."""
    last = bars["BTCUSDT"].index[-1]
    live = {s: df for s, df in bars.items() if len(df) >= 60 and df.index[-1] == last}
    ms = build_state(live, C.REGIME_N)
    cols, sigs, _, _ = build_columns(live, configured_experts(), ms)
    t = int(last.value)
    plan = {k: pd.DataFrame({"long_entry": [bool(sg.long_entry[-1])], "long_stop": [float(sg.long_stop[-1])],
                             "long_trail": [float(sg.long_trail[-1])], "long_exit": [bool(sg.long_exit[-1])]},
                            index=[t]) for k, sg in sigs.items()}
    snap = ms.snapshot(last)
    state = dict(bar_open=str(last), btc_bull=snap["btc_bull"], breadth=snap["breadth"],
                 btc_vol_pct=snap["btc_vol_pct"],
                 experts={ex.name: [s for s in live if bool(ex.requires(ms.for_symbol(s), **ex.req_params)
                                                              .fillna(False).iloc[-1])]
                          for ex in configured_experts()})
    return plan, state


class EnsembleLiveStrategy(EnsembleStrategy):
    MAX_DECISION_DELAY = pd.Timedelta(minutes=30)  # never act on a bar that closed longer ago than this

    def __init__(self, config: EnsembleConfig, mode: str):
        super().__init__(config, plan={}, instruments={}, bar_types={}, expected={})
        self.mode = mode
        paper_only(mode, config.risk)
        self.report = PaperReport()
        self.manifests = {e.name: expert_manifest(e) for e in configured_experts()}
        self.report.event("experts", manifests=self.manifests, status="PAPER_OBSERVATION_ONLY")
        self.last_market_state = None
        self.last_mark_ns = {}
        self._decision_task = None
        self.loop = None
        self.halted = False
        self.held_path = STATE_DIR / f"nautilus_{mode}_held.json"
        # Sandbox has no durable exchange account. Never adopt old owners into a fresh balance.
        self.done_bar = None

    # ------------------------------------------------------------------ lifecycle
    def on_start(self):
        for s in C.UNIVERSE:
            inst = self.cache.instrument(instrument_id(s))
            if inst is None:
                self.log.error(f"instrument {s} not loaded; stopping")
                self.stop()
                return
            self.inst[s] = inst
            bt = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")  # prices for sandbox fills + marks
            self.bar_types[s] = bt
            self.subscribe_bars(bt)
        self._adopt()
        self.clock.set_timer("tycho_decide", pd.Timedelta(minutes=1), callback=self._tick)
        self.clock.set_timer("tycho_report", pd.Timedelta(minutes=1), callback=self._report_tick)

    def on_bar(self, bar):
        sym = bar.bar_type.instrument_id.symbol.value.replace("-PERP", "")
        self.close[sym] = float(bar.close)
        self.last_mark_ns[sym] = bar.ts_event

    def _report_tick(self, _event):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._report_now)
        else:
            self._report_now()

    def _report_now(self):
        try:
            if not getattr(self, "_stopping", False):
                self._check_risk()
            positions = []
            for p in self.cache.positions_open(venue=VENUE):
                s = p.instrument_id.symbol.value.replace("-PERP", "")
                mark = self.close.get(s)
                positions.append(dict(symbol=s, expert=self.held.get(s, {}).get("col"),
                                      quantity=float(p.signed_qty), entry=float(p.avg_px_open), mark=mark,
                                      unrealized_pnl=float(p.unrealized_pnl(self.inst[s].make_price(mark)))
                                      if mark is not None else None))
            row = self.report.snapshot(super()._equity(), positions, last_decided_bar=str(self.done_bar),
                                       market_state=self.last_market_state, manifests=self.manifests,
                                       realized_pnl_after_fees=float(self.portfolio.account(VENUE).balance_total(USDT)) - self.report.initial,
                                       unrealized_pnl=sum(p["unrealized_pnl"] or 0 for p in positions),
                                       status="STOPPED" if getattr(self, "_stopping", False) else "HALTED" if self.halted else "RUNNING",
                                       funding_model="Binance settled funding rates x settlement mark x simulated inventory")
            self.log.info("PAPER_ACCOUNT " + json.dumps(row, default=str))
        except Exception as exc:
            self.report.event("report_error", error=repr(exc))

    def _equity(self):
        # Cost reserves reduce sizing as well as reported equity; Nautilus balances stay native.
        return super()._equity() - self.report.slippage - self.report.funding

    def _check_risk(self):
        if self._equity() <= self.report.peak * .80:
            self.halted = True
        if self.halted:
            for s in list(self.held):
                self._exit(s, "safety_halt")
        for p in self.cache.positions_open(venue=VENUE):
            s = p.instrument_id.symbol.value.replace("-PERP", "")
            h = self.held.get(s)
            if h is None:
                self.halted = True
                self.cancel_all_orders(p.instrument_id)
                self.close_position(p)
                self.report.event("safety_halt", reason="unowned position", symbol=s)
            elif not h.get("exiting"):
                stop = self.cache.order(h["stop_order"]) if h["stop_order"] is not None else None
                if stop is None or not stop.is_open:
                    self.halted = True
                    self.report.event("safety_halt", reason="unprotected position", symbol=s)
                    self._exit(s, "unprotected")

    def _tick(self, _event):
        # Nautilus LiveClock dispatches timer callbacks on a native thread.
        self.loop.call_soon_threadsafe(self._schedule_tick)

    def _schedule_tick(self):
        if self._decision_task is None or self._decision_task.done():
            self._decision_task = asyncio.get_running_loop().create_task(self._tick_async())

    async def _tick_async(self):
        try:
            await self._tick_inner()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never let a data hiccup kill the node; retried next minute
            self.log.error(f"decision tick failed: {e!r}")
            self.report.event("abstain", reason=repr(e))

    async def _tick_inner(self):
        now = await asyncio.to_thread(server_now)
        # Network work stays outside the Nautilus event loop; fills/stops keep processing.
        for symbol in sorted({s for s, _, _ in self.report.inventory_events}):
            cursor = max((t + 1 for s, t in self.report.funding_events if s == symbol), default=self.report.started_ms)
            rows = await asyncio.to_thread(_get, "/fapi/v1/fundingRate",
                                            {"symbol": symbol, "startTime": cursor, "limit": 1000})
            self.report.settle_funding(symbol, rows)
        self._check_risk()
        cur = now.floor(C.TIMEFRAME)
        if cur == self.done_bar or now < cur + pd.Timedelta(seconds=15):
            return
        if now > cur + self.MAX_DECISION_DELAY:  # e.g. restarted mid-bar: wait for the next close
            self.log.warning(f"bar closed at {cur} is older than {self.MAX_DECISION_DELAY}; skipping it")
            if self.last_market_state is None:
                bars = await asyncio.to_thread(fetch_all_bars)
                validate_bars(bars, cur - BAR)
                plan, self.last_market_state = latest_plan(bars)
                self.report.event("observation", reason="stale decision window; no orders",
                                  state=self.last_market_state,
                                  signals={k: {f: v.iloc[0] for f, v in df.items()} for k, df in plan.items()})
            self.report.event("abstain", reason="closed bar exceeds 30 minute decision delay", bar_close=str(cur))
            self.done_bar = cur
            self._persist()
            return
        # Fetch in a worker while the node continues to process prices and stop fills.
        bars = await asyncio.to_thread(fetch_all_bars)
        validate_bars(bars, cur - BAR)
        if {e.name: expert_manifest(e) for e in configured_experts()} != self.manifests:
            raise ValueError("expert parameters/code changed during session; abstain")
        if bars["BTCUSDT"].index[-1] != cur - BAR:
            self.log.warning(f"bars not ready: have {bars['BTCUSDT'].index[-1]} want {cur - BAR}")
            return
        self.plan, state = latest_plan(bars)
        self.last_market_state = state
        self.cols = list(self.plan)
        prices = await asyncio.to_thread(lambda: {s: last_price(s) for s in C.UNIVERSE})
        after = await asyncio.to_thread(server_now)
        if after > cur + self.MAX_DECISION_DELAY or after.floor(C.TIMEFRAME) != cur:
            raise ValueError("data preparation crossed decision deadline; abstain")
        stale = [s for s in C.UNIVERSE if after.value - self.last_mark_ns.get(s, 0) > pd.Timedelta(minutes=3).value]
        if stale:
            raise ValueError(f"stream prices are missing/stale: {stale}; abstain")
        for s in C.UNIVERSE:  # fresh prices for sizing (not the 4h close)
            self.close[s] = prices[s]
            if not 0 < self.close[s] < float("inf"):
                raise ValueError(f"invalid price for {s}")
        self.report.event("decision", bar_open=str(cur - BAR), state=state,
                          signals={k: {f: v.iloc[0] for f, v in df.items()} for k, df in self.plan.items()})
        if self.halted or (STATE_DIR / "PAUSE").exists():
            for frame in self.plan.values():
                frame["long_entry"] = False
            self.report.event("entry_blocked", reason="drawdown halt or live_state/PAUSE")
        t_open = int((cur - BAR).value)
        self._decide(t_open)
        self.done_bar = cur
        self._persist()
        self.log.info("MARKET_STATE " + json.dumps(state, default=str))
        self.log.info("HELD " + json.dumps({s: {k: str(v) for k, v in h.items()} for s, h in self.held.items()}))

    def _exit(self, s, reason):
        self.report.event("exit_decision", symbol=s, reason=reason)
        super()._exit(s, reason)

    def on_stop(self):
        self._stopping = True
        if self._decision_task is not None:
            self._decision_task.cancel()
        self._report_now()
        self.report.event("session_stop", note="Sandbox session ended; next start is a fresh account")

    # ------------------------------------------------------------------ persistence + restart safety
    def _load(self) -> dict:
        try:
            return json.loads(self.held_path.read_text()) if self.held_path.exists() else {}
        except json.JSONDecodeError:
            return {}

    def _persist(self):
        STATE_DIR.mkdir(exist_ok=True)
        tmp = self.held_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done_bar": str(self.done_bar) if self.done_bar is not None else None,
                                   "held": {s: dict(col=h["col"], stop=h["stop"]) for s, h in self.held.items()}}))
        tmp.replace(self.held_path)

    def _adopt(self):
        """Every open position must be owned and protected, else cancel its orders and flatten it."""
        saved = self._load().get("held", {})
        for pos in self.cache.positions_open(venue=VENUE):
            s = pos.instrument_id.symbol.value.replace("-PERP", "")
            stops = [o for o in self.cache.orders_open(instrument_id=pos.instrument_id)
                     if o.order_type == OrderType.STOP_MARKET and o.side == OrderSide.SELL]
            if s in saved and len(stops) == 1:
                self.held[s] = dict(col=saved[s]["col"], stop=float(stops[0].trigger_price),
                                    stop_order=stops[0].client_order_id, entry_order=None)
                self.log.info(f"adopted {s} owned by {saved[s]['col']}")
            else:
                self.log.warning(f"unowned/unprotected position {s} ({len(stops)} stops); flattening")
                self.cancel_all_orders(pos.instrument_id)
                self.close_position(pos)
        self._persist()

    def on_order_filled(self, event):
        self.report.event("order_fill_owner", order=str(event.client_order_id),
                          expert=self.held.get(self._sym(event), {}).get("col"))
        self.report.fill(event)
        super().on_order_filled(event)
        s = self._sym(event)
        if s not in self.held and float(self.portfolio.net_position(event.instrument_id)) != 0:
            self.halted = True
            self.report.event("safety_halt", reason="unowned fill; flattening", symbol=s)
            self.cancel_all_orders(event.instrument_id)
            for p in self.cache.positions_open(instrument_id=event.instrument_id):
                self.close_position(p)
        self._persist()

    def _order_failed(self, event, what: str):
        self.report.event("order_failure", symbol=self._sym(event), order=str(event.client_order_id), reason=what)
        super()._order_failed(event, what)
        self._persist()


def build_node(mode: str, risk: float) -> TradingNode:
    paper_only(mode, risk)
    ids = frozenset(str(instrument_id(s)) for s in C.UNIVERSE)
    data = BinanceDataClientConfig(account_type=BinanceAccountType.USDT_FUTURES, environment=BinanceEnvironment.LIVE,
                                   instrument_provider=InstrumentProviderConfig(load_ids=ids))
    if mode in {"sandbox", "paper"}:
        exec_cfg = SandboxExecutionClientConfig(venue="BINANCE", starting_balances=["10000 USDT"], base_currency="USDT",
                                                oms_type="NETTING", account_type="MARGIN",
                                                default_leverage=Decimal(10), bar_execution=True)
        exec_factory = SandboxLiveExecClientFactory
    node = TradingNode(TradingNodeConfig(
        trader_id="TYCHO-001", logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=True),
        data_clients={"BINANCE": data}, exec_clients={"BINANCE": exec_cfg}))
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE", exec_factory)
    claims = [str(instrument_id(s)) for s in C.UNIVERSE]  # own reconciled orders after a restart
    strategy = EnsembleLiveStrategy(EnsembleConfig(risk=risk, external_order_claims=claims), mode)
    strategy.loop = node.get_event_loop()
    node.trader.add_strategy(strategy)
    node.build()
    return node
