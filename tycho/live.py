"""Live/paper execution of the expert ensemble on Binance USDT-M perpetuals.

Each cycle builds the MarketState from closed 4h bars; every configured expert requests it and
only acts where its required state holds. `decide` is a pure function of bars + positions and is
replayed against the backtest engine in tests (tests/test_live.py) so live == backtest rules.

Modes:
  paper   : public market data only, simulated fills/stops, no keys needed (default)
  testnet : Binance demo-trading account via ccxt (BINANCE_API_KEY/SECRET of the DEMO account)
  live    : real money. Requires keys AND --i-understand-real-money. Exchange-side stop orders
            protect positions even if this process dies.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import _get
from .engine import Costs
from .ensemble import configured_experts
from .marketstate import build_state
from .portfolio import MAJORS

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "live_state"
JOURNAL_DIR = ROOT / "journal"
BAR = pd.Timedelta(C.TIMEFRAME)


# ----------------------------------------------------------------------------- decision logic
@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    stop: float
    r_unit: float
    entry_time: str
    stop_order_id: str | None = None
    expert: str = "trend_long"   # owner; one expert per coin (Binance one-way netting)


@dataclass
class Decision:
    bar_time: pd.Timestamp                            # close time of the bar decided on
    stop_moves: dict = field(default_factory=dict)    # symbol -> new stop
    exits: list = field(default_factory=list)         # symbols to close at market (expert exit signal)
    entries: list = field(default_factory=list)       # [(expert, symbol, initial_stop)] in priority order
    regime_bull: bool = False
    state: dict = field(default_factory=dict)         # MarketState snapshot + which experts are active


def decide(bars: dict[str, pd.DataFrame], positions: dict[str, Position]) -> Decision:
    """bars: CLOSED 4h bars per symbol, all ending at the same last bar."""
    last = bars["BTCUSDT"].index[-1]
    live = {s: df for s, df in bars.items() if len(df) >= 60 and df.index[-1] == last}
    ms = build_state(live, C.REGIME_N)
    d = Decision(bar_time=last + BAR, regime_bull=bool(ms.btc_bull.iloc[-1]))
    snap = ms.snapshot(last)
    d.state = dict(btc_bull=snap["btc_bull"], breadth=snap["breadth"], btc_vol_pct=snap["btc_vol_pct"], experts={})
    experts = configured_experts()
    sigs = {(ex.name, s): ex.signals(live[s], ms.for_symbol(s)) for ex in experts for s in C.UNIVERSE if s in live}
    # pass 1: manage held coins (owner's exit signal, else trailing stop)
    for s, p in positions.items():
        sig = sigs.get((p.expert, s))
        if sig is None:
            continue
        if sig.long_exit[-1]:
            d.exits.append(s)
        elif np.isfinite(sig.long_trail[-1]) and sig.long_trail[-1] > p.stop:
            d.stop_moves[s] = float(sig.long_trail[-1])
    # pass 2: entries in priority order; a coin exiting this bar may be re-entered (as the engine does)
    for ex in experts:
        active = []
        for s in C.UNIVERSE:
            if s not in live:
                continue
            st = ms.for_symbol(s)
            if bool(ex.requires(st, **ex.req_params).fillna(False).iloc[-1]):
                active.append(s)
            sig = sigs[(ex.name, s)]
            if (s not in positions or s in d.exits) and sig.long_entry[-1] and np.isfinite(sig.long_stop[-1]):
                d.entries.append((ex.name, s, float(sig.long_stop[-1])))
        d.state["experts"][ex.name] = active  # coins where this expert's requested state holds
    return d


def size(equity: float, gross_now: float, px: float, stop: float, risk: float) -> float:
    dist = px - stop
    if dist <= 0:
        return 0.0
    room = max(equity * C.MAX_GROSS - gross_now, 0.0)
    return max(min(equity * risk / dist, equity * C.MAX_LEV_POS / px, room / px), 0.0)


# ----------------------------------------------------------------------------- market data
def fetch_closed_bars(symbol: str, n: int) -> pd.DataFrame:
    rows: list = []
    end = None
    while len(rows) < n + 1:
        params = {"symbol": symbol, "interval": C.TIMEFRAME, "limit": min(1500, n + 1 - len(rows))}
        if end is not None:
            params["endTime"] = end
        batch = _get("/fapi/v1/klines", params)
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < params["limit"]:
            break
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "ct", "qvol", "n", "a", "b", "x"])
    df = df.drop_duplicates("t")
    closed = df["ct"].astype("int64") < int(server_now().timestamp() * 1000)  # exchange clock, not local
    df = df[closed]
    out = df[["open", "high", "low", "close", "volume", "qvol"]].astype(float)
    out.index = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    return out.sort_index().iloc[-n:]


def fetch_all_bars() -> dict[str, pd.DataFrame]:
    out = {}
    for s in C.UNIVERSE:
        out[s] = fetch_closed_bars(s, C.REGIME_WARMUP_BARS if s == "BTCUSDT" else C.WARMUP_BARS)
    return out


def last_price(symbol: str) -> float:
    return float(_get("/fapi/v1/ticker/price", {"symbol": symbol})["price"])


# ----------------------------------------------------------------------------- brokers
class PaperBroker:
    """Simulated account: market fills at last price +/- slippage, taker fees, stops checked on poll."""

    def __init__(self, state: dict):
        self.state = state
        self.state.setdefault("cash", 10_000.0)

    def costs(self, s):
        return Costs(slippage=0.0002 if s in MAJORS else 0.0005)

    def equity(self, positions, prices) -> float:
        return self.state["cash"] + sum(p.qty * (prices[s] - p.entry) for s, p in positions.items())

    def buy(self, s, qty, ref_px):
        c = self.costs(s)
        px = ref_px * (1 + c.slippage)
        self.state["cash"] -= qty * px * c.fee
        return px, qty

    def sell(self, p: Position, ref_px):
        c = self.costs(p.symbol)
        px = ref_px * (1 - c.slippage)
        self.state["cash"] += p.qty * (px - p.entry) - p.qty * px * c.fee
        return px

    def place_stop(self, p: Position, stop: float):
        return None

    def check_stops(self, positions, prices):
        hit = []
        for s, p in positions.items():
            if prices[s] <= p.stop:
                hit.append((s, self.sell(p, prices[s])))  # stop-market fills at market
        return hit


class BinanceBroker:
    """ccxt binanceusdm. One-way mode, cross margin. Stops are exchange-side reduce-only orders."""

    def __init__(self, testnet: bool):
        raise RuntimeError("Exchange execution disabled. Use python -m scripts.nautilus_live")
        import ccxt
        key, sec = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
        if not key or not sec:
            raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")
        self.x = ccxt.binanceusdm({"apiKey": key, "secret": sec, "enableRateLimit": True})
        if testnet:
            self.x.enable_demo_trading(True)
        self.x.load_markets()

    @staticmethod
    def ccxt_sym(s):
        return f"{s[:-4]}/USDT:USDT"

    def equity(self, positions=None, prices=None) -> float:
        b = self.x.fetch_balance()
        return float(b["info"]["totalMarginBalance"])

    def exchange_positions(self) -> dict[str, float]:
        out = {}
        for p in self.x.fetch_positions([self.ccxt_sym(s) for s in C.UNIVERSE]):
            amt = float(p["info"].get("positionAmt", 0) or 0)
            if amt != 0:
                out[p["info"]["symbol"]] = amt
        return out

    def buy(self, s, qty, ref_px):
        sym = self.ccxt_sym(s)
        self.x.set_leverage(int(np.ceil(C.MAX_GROSS)) + 1, sym)
        q = float(self.x.amount_to_precision(sym, qty))
        mkt = self.x.market(sym)
        min_cost = (mkt.get("limits", {}).get("cost", {}) or {}).get("min") or 5.0
        if q <= 0 or q * ref_px < min_cost:
            return None, 0.0
        o = self.x.create_order(sym, "market", "buy", q)
        px = float(o.get("average") or o.get("price") or ref_px)
        return px, float(o.get("filled") or q)

    def sell(self, p: Position, ref_px):
        sym = self.ccxt_sym(p.symbol)
        o = self.x.create_order(sym, "market", "sell", p.qty, params={"reduceOnly": True})
        self.cancel_order(p.symbol, p.stop_order_id)
        return float(o.get("average") or o.get("price") or ref_px)

    def place_stop(self, p: Position, stop: float) -> str:
        """Create the NEW stop first, cancel the old one only after it exists (never unprotected).
        Trigger on last trade price (CONTRACT_PRICE) to match the backtest's use of bar lows."""
        sym = self.ccxt_sym(p.symbol)
        o = self.x.create_order(sym, "market", "sell", p.qty,
                                params={"stopLossPrice": float(self.x.price_to_precision(sym, stop)),
                                        "reduceOnly": True, "workingType": "CONTRACT_PRICE"})
        self.cancel_order(p.symbol, p.stop_order_id)
        return str(o["id"])

    def cancel_order(self, symbol, order_id):
        if order_id:
            try:
                self.x.cancel_order(order_id, self.ccxt_sym(symbol), params={"stop": True})
            except Exception as e:  # already triggered/cancelled
                log({"event": "cancel_stop_failed", "symbol": symbol, "error": str(e)})

    def check_stops(self, positions, prices):
        """Exchange executes stops; detect positions that disappeared."""
        live = self.exchange_positions()
        return [(s, prices[s]) for s in positions if s not in live]  # exit px is an estimate (last price)

    def reconcile(self, positions) -> list[str]:
        live = self.exchange_positions()
        problems = [f"untracked exchange position {s} qty={live[s]}" for s in set(live) - set(positions)]
        problems += [f"qty mismatch {s}: state={p.qty} exchange={live[s]}"
                     for s, p in positions.items() if s in live and abs(live[s] - p.qty) > 1e-9]
        return problems


# ----------------------------------------------------------------------------- state + journal
def state_path(mode):
    return STATE_DIR / f"{mode}_state.json"


def load_state(mode) -> tuple[dict, dict[str, Position]]:
    p = state_path(mode)
    if not p.exists():
        return {}, {}
    raw = json.loads(p.read_text())
    return raw.get("account", {}), {s: Position(**v) for s, v in raw.get("positions", {}).items()}


def save_state(mode, account, positions):
    STATE_DIR.mkdir(exist_ok=True)
    tmp = state_path(mode).with_suffix(".tmp")
    tmp.write_text(json.dumps({"account": account, "positions": {s: asdict(p) for s, p in positions.items()}}, indent=2))
    tmp.replace(state_path(mode))


_MODE = ["paper"]


def log(rec: dict):
    JOURNAL_DIR.mkdir(exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "mode": _MODE[0]} | rec
    with (JOURNAL_DIR / f"{_MODE[0]}_{datetime.now(timezone.utc):%Y%m}.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec, default=str), flush=True)


# ----------------------------------------------------------------------------- loop
def server_now() -> pd.Timestamp:
    return pd.Timestamp(int(_get("/fapi/v1/time", {})["serverTime"]), unit="ms", tz="UTC")


def _stopped(s, px, p, account, positions, mode):
    log({"event": "stopped", "symbol": s, "exit_est": px, "entry": p.entry, "qty": p.qty,
         "r": (px - p.entry) * p.qty / p.r_unit})
    save_state(mode, account, positions)


def cycle(mode: str, broker, account: dict, positions: dict[str, Position], risk: float) -> bool:
    """One decision cycle after a 4h bar closes. Returns False if the latest closed bar isn't
    available yet (caller retries). Idempotent per bar: a restart never re-acts on a processed bar."""
    expected_last = server_now().floor(C.TIMEFRAME) - BAR  # open time of the last CLOSED bar
    bars = fetch_all_bars()
    if bars["BTCUSDT"].index[-1] != expected_last:
        log({"event": "bars_not_ready", "have": str(bars["BTCUSDT"].index[-1]), "want": str(expected_last)})
        return False
    prices = {s: last_price(s) for s in C.UNIVERSE}
    for s, px in broker.check_stops(positions, prices):
        _stopped(s, px, positions.pop(s), account, positions, mode)
    dec = decide(bars, positions)
    if account.get("last_bar") == str(dec.bar_time):
        log({"event": "bar_already_processed", "bar": str(dec.bar_time)})
        return True
    # ratchet trailing stops; on failure the old stop stays live and the move is retried next bar
    for s, new_stop in dec.stop_moves.items():
        p = positions[s]
        try:
            p.stop_order_id = broker.place_stop(p, new_stop)
            p.stop = new_stop
            log({"event": "stop_moved", "symbol": s, "stop": new_stop})
        except Exception as e:
            log({"event": "stop_move_failed", "symbol": s, "stop": new_stop, "kept": p.stop, "error": repr(e)})
        save_state(mode, account, positions)
    for s in dec.exits:
        p = positions[s]
        exit_px = broker.sell(p, prices[s])  # sell first: if it raises, the position stays tracked
        positions.pop(s)
        log({"event": "exit_signal", "symbol": s, "expert": p.expert, "exit": exit_px, "entry": p.entry,
             "r": (exit_px - p.entry) * p.qty / p.r_unit})
        save_state(mode, account, positions)
    for expert, s, stop in dec.entries:
        if s in positions:
            continue  # coin already owned by a higher-priority expert
        if len(positions) >= C.MAX_POSITIONS:
            log({"event": "entry_skipped_full", "symbol": s})
            continue
        eq = broker.equity(positions, prices)
        gross = sum(abs(p.qty) * prices[x] for x, p in positions.items())
        est_px = prices[s] * (1 + (0.0002 if s in MAJORS else 0.0005))  # same estimate as backtest
        qty = size(eq, gross, est_px, stop, risk)
        if qty <= 0:
            log({"event": "entry_skipped_size", "symbol": s, "price": prices[s], "stop": stop})
            continue
        px, filled = broker.buy(s, qty, prices[s])
        if not filled:
            log({"event": "entry_skipped_min_notional", "symbol": s, "qty": qty})
            continue
        p = Position(s, filled, px, stop, filled * (px - stop), str(dec.bar_time), expert=expert)
        positions[s] = p
        save_state(mode, account, positions)  # persist the fill before anything else can fail
        log({"event": "entry", "symbol": s, "expert": expert, "qty": filled, "price": px, "stop": stop, "equity": eq})
        try:
            p.stop_order_id = broker.place_stop(p, stop)
        except Exception as e:  # never hold an unprotected position: flatten it
            log({"event": "entry_stop_failed_flattening", "symbol": s, "error": repr(e)})
            exit_px = broker.sell(p, prices[s])
            positions.pop(s)
            log({"event": "flattened", "symbol": s, "exit": exit_px})
        save_state(mode, account, positions)
    account["last_bar"] = str(dec.bar_time)
    eq = broker.equity(positions, prices)
    log({"event": "cycle", "bar": str(dec.bar_time), "regime_bull": dec.regime_bull, "equity": eq,
         "market_state": dec.state,
         "open": {s: dict(expert=p.expert, qty=p.qty, entry=p.entry, stop=p.stop, upnl=p.qty * (prices[s] - p.entry))
                  for s, p in positions.items()}})
    save_state(mode, account, positions)
    return True


def main(mode: str = "paper", risk: float = C.RISK_DEFAULT, once: bool = False, poll: int = 60):
    raise RuntimeError("Legacy runner retired. Use python -m scripts.nautilus_live for Nautilus paper trading")
    _MODE[0] = mode
    account, positions = load_state(mode)
    broker = PaperBroker(account) if mode == "paper" else BinanceBroker(testnet=(mode == "testnet"))
    if hasattr(broker, "reconcile"):
        problems = broker.reconcile(positions)
        if problems:
            for msg in problems:
                log({"event": "reconcile_problem", "detail": msg})
            raise SystemExit("state and exchange disagree; resolve manually (see journal) before starting")
    log({"event": "start", "risk": risk, "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
         "positions": list(positions)})
    done_bar = None
    while True:
        try:
            now = server_now()
            cur = now.floor(C.TIMEFRAME)
            if cur != done_bar and now >= cur + pd.Timedelta(seconds=15):
                if cycle(mode, broker, account, positions, risk):
                    done_bar = cur
                if once and done_bar is not None:
                    return
            elif mode == "paper" and positions:  # paper: emulate resting stop orders between bars
                prices = {s: last_price(s) for s in positions}
                for s, px in broker.check_stops(positions, prices):
                    _stopped(s, px, positions.pop(s), account, positions, mode)
        except Exception as e:
            log({"event": "loop_error", "error": repr(e)})
            if once:
                raise
        time.sleep(poll)
