"""NautilusTrader live node for the expert ensemble.

Modes:
  sandbox : Binance public market data (no keys) + Nautilus simulated execution (paper)
  testnet : Binance DEMO futures account (BINANCE_API_KEY/SECRET of the demo account)
  live    : real money (keys + --i-understand-real-money)

Decision path: a 1-minute timer checks the exchange clock; once a 4h bar has closed it fetches
closed bars over REST (tycho.live.fetch_all_bars, same path as the tested tycho runner), builds the
MarketState, lets each configured expert compute its latest signal, and runs the SAME `_decide`
code that the Nautilus backtest executed (tycho.nautilus_bt.EnsembleStrategy).
Expert ownership of positions is persisted to live_state/nautilus_<mode>_held.json.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal

import pandas as pd
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig, BinanceExecClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory, BinanceLiveExecClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig, LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId

from . import config as C
from .ensemble import build_columns, configured_experts
from .live import STATE_DIR, fetch_all_bars, last_price, server_now
from .marketstate import build_state
from .nautilus_bt import VENUE, EnsembleConfig, EnsembleStrategy

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
        self.held_path = STATE_DIR / f"nautilus_{mode}_held.json"
        saved = self._load()
        self.done_bar = pd.Timestamp(saved["done_bar"]) if saved.get("done_bar") else None

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

    def on_bar(self, bar):
        sym = bar.bar_type.instrument_id.symbol.value.replace("-PERP", "")
        self.close[sym] = float(bar.close)

    def _tick(self, _event):
        try:
            self._tick_inner()
        except Exception as e:  # never let a data hiccup kill the node; retried next minute
            self.log.error(f"decision tick failed: {e!r}")

    def _tick_inner(self):
        now = server_now()
        cur = now.floor(C.TIMEFRAME)
        if cur == self.done_bar or now < cur + pd.Timedelta(seconds=15):
            return
        if now > cur + self.MAX_DECISION_DELAY:  # e.g. restarted mid-bar: wait for the next close
            self.log.warning(f"bar closed at {cur} is older than {self.MAX_DECISION_DELAY}; skipping it")
            self.done_bar = cur
            self._persist()
            return
        # NOTE: blocking REST (~15-30 s once per 4h) inside the event loop; reduce-only stops rest on
        # the exchange meanwhile.
        bars = fetch_all_bars()
        if bars["BTCUSDT"].index[-1] != cur - BAR:
            self.log.warning(f"bars not ready: have {bars['BTCUSDT'].index[-1]} want {cur - BAR}")
            return
        self.plan, state = latest_plan(bars)
        self.cols = list(self.plan)
        for s in C.UNIVERSE:  # fresh prices for sizing (not the 4h close)
            self.close[s] = last_price(s)
        t_open = int((cur - BAR).value)
        self._decide(t_open)
        self.done_bar = cur
        self._persist()
        self.log.info("MARKET_STATE " + json.dumps(state, default=str))
        self.log.info("HELD " + json.dumps({s: {k: str(v) for k, v in h.items()} for s, h in self.held.items()}))

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
        super().on_order_filled(event)
        self._persist()

    def _order_failed(self, event, what: str):
        super()._order_failed(event, what)
        self._persist()


def build_node(mode: str, risk: float) -> TradingNode:
    ids = frozenset(str(instrument_id(s)) for s in C.UNIVERSE)
    key, sec = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
    data = BinanceDataClientConfig(account_type=BinanceAccountType.USDT_FUTURES, environment=BinanceEnvironment.LIVE,
                                   instrument_provider=InstrumentProviderConfig(load_ids=ids))
    if mode == "sandbox":
        exec_cfg = SandboxExecutionClientConfig(venue="BINANCE", starting_balances=["10000 USDT"], base_currency="USDT",
                                                oms_type="NETTING", account_type="MARGIN",
                                                default_leverage=Decimal(10), bar_execution=True)
        exec_factory = SandboxLiveExecClientFactory
    else:
        if not key or not sec:
            raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")
        env = BinanceEnvironment.DEMO if mode == "testnet" else BinanceEnvironment.LIVE
        exec_cfg = BinanceExecClientConfig(api_key=key, api_secret=sec, account_type=BinanceAccountType.USDT_FUTURES,
                                           environment=env, instrument_provider=InstrumentProviderConfig(load_ids=ids),
                                           use_reduce_only=True)
        exec_factory = BinanceLiveExecClientFactory
    node = TradingNode(TradingNodeConfig(
        trader_id="TYCHO-001", logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=True),
        data_clients={"BINANCE": data}, exec_clients={"BINANCE": exec_cfg}))
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE", exec_factory)
    claims = [str(instrument_id(s)) for s in C.UNIVERSE]  # own reconciled orders after a restart
    node.trader.add_strategy(EnsembleLiveStrategy(EnsembleConfig(risk=risk, external_order_claims=claims), mode))
    node.build()
    return node
