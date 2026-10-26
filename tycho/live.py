"""Live/paper execution of the configured strategy on Binance USDT-M perpetuals.

The decision logic (`decide`) is a pure function of closed 4h bars + current positions, and is
replayed against the backtest engine in tests (see tests/test_live.py) so live == backtest rules.

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
from .portfolio import MAJORS
from .strategies import STRATEGIES, ema

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


@dataclass
class Decision:
    bar_time: pd.Timestamp                      # close time of the bar decided on
    stop_moves: dict = field(default_factory=dict)   # symbol -> new stop
    entries: list = field(default_factory=list)      # [(symbol, initial_stop)] in priority order
    regime_bull: bool = False


def decide(bars: dict[str, pd.DataFrame], positions: dict[str, Position]) -> Decision:
    """bars: CLOSED 4h bars per symbol, all ending at the same last bar."""
    btc = bars["BTCUSDT"]
    bull = bool(btc["close"].iloc[-1] > ema(btc["close"], C.REGIME_N).iloc[-1])
    last = btc.index[-1]
    d = Decision(bar_time=last + BAR, regime_bull=bull)
    fn = STRATEGIES[C.STRATEGY]
    for s in C.UNIVERSE:
        df = bars.get(s)
        if df is None or len(df) < 60 or df.index[-1] != last:
            continue
        sig = fn(df.iloc[-C.WARMUP_BARS:], **C.PARAMS)
        if s in positions:
            trail = sig.long_trail[-1]
            if np.isfinite(trail) and trail > positions[s].stop:
                d.stop_moves[s] = float(trail)
        elif bull and sig.long_entry[-1] and np.isfinite(sig.long_stop[-1]):
            d.entries.append((s, float(sig.long_stop[-1])))
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
    closed = df["ct"].astype("int64") < int(time.time() * 1000)
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

    def place_stop(self, p: Position):
        return None

    def check_stops(self, positions, prices):
        hit = []
        for s, p in positions.items():
            if prices[s] <= p.stop:
                hit.append((s, self.sell(p, min(prices[s], p.stop))))
        return hit


class BinanceBroker:
    """ccxt binanceusdm. One-way mode, cross margin. Stops are exchange-side reduce-only orders."""

    def __init__(self, testnet: bool):
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
        self.cancel_stop(p)
        o = self.x.create_order(sym, "market", "sell", p.qty, params={"reduceOnly": True})
        return float(o.get("average") or o.get("price") or ref_px)

    def place_stop(self, p: Position):
        sym = self.ccxt_sym(p.symbol)
        self.cancel_stop(p)
        stop = float(self.x.price_to_precision(sym, p.stop))
        o = self.x.create_order(sym, "market", "sell", p.qty,
                                params={"stopLossPrice": stop, "reduceOnly": True, "workingType": "MARK_PRICE"})
        return str(o["id"])

    def cancel_stop(self, p: Position):
        if p.stop_order_id:
            try:
                self.x.cancel_order(p.stop_order_id, self.ccxt_sym(p.symbol), params={"stop": True})
            except Exception as e:  # already filled/cancelled
                log({"event": "cancel_stop_failed", "symbol": p.symbol, "error": str(e)})
            p.stop_order_id = None

    def check_stops(self, positions, prices):
        """Exchange executes stops; detect positions that disappeared."""
        live = self.exchange_positions()
        return [(s, prices[s]) for s in positions if s not in live]  # exit px is an estimate (last price)

    def reconcile(self, positions):
        live = self.exchange_positions()
        for s in set(live) - set(positions):
            log({"event": "warn_untracked_exchange_position", "symbol": s, "qty": live[s]})
        for s, p in positions.items():
            if s in live and abs(live[s] - p.qty) > 1e-9:
                log({"event": "warn_qty_mismatch", "symbol": s, "state": p.qty, "exchange": live[s]})


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
def cycle(mode: str, broker, account: dict, positions: dict[str, Position], risk: float):
    """One decision cycle right after a 4h bar closes."""
    bars = fetch_all_bars()
    prices = {s: last_price(s) for s in C.UNIVERSE}
    # 1) positions closed by stops since last cycle
    for s, px in broker.check_stops(positions, prices):
        p = positions.pop(s)
        log({"event": "stopped", "symbol": s, "exit": px, "entry": p.entry, "qty": p.qty,
             "r": (px - p.entry) * p.qty / p.r_unit})
    dec = decide(bars, positions)
    # 2) ratchet trailing stops
    for s, new_stop in dec.stop_moves.items():
        positions[s].stop = new_stop
        positions[s].stop_order_id = broker.place_stop(positions[s])
        log({"event": "stop_moved", "symbol": s, "stop": new_stop})
    # 3) entries
    for s, stop in dec.entries:
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
        p = Position(s, filled, px, stop, filled * (px - stop), str(dec.bar_time))
        p.stop_order_id = broker.place_stop(p)
        positions[s] = p
        log({"event": "entry", "symbol": s, "qty": filled, "price": px, "stop": stop, "equity": eq})
    eq = broker.equity(positions, prices)
    log({"event": "cycle", "bar": str(dec.bar_time), "regime_bull": dec.regime_bull, "equity": eq,
         "open": {s: dict(qty=p.qty, entry=p.entry, stop=p.stop, upnl=p.qty * (prices[s] - p.entry))
                  for s, p in positions.items()}})
    save_state(mode, account, positions)


def next_close(now: pd.Timestamp) -> pd.Timestamp:
    return now.floor(C.TIMEFRAME) + BAR


def main(mode: str = "paper", risk: float = C.RISK_DEFAULT, once: bool = False, poll: int = 60):
    _MODE[0] = mode
    account, positions = load_state(mode)
    broker = PaperBroker(account) if mode == "paper" else BinanceBroker(testnet=(mode == "testnet"))
    if hasattr(broker, "reconcile"):
        broker.reconcile(positions)
    log({"event": "start", "risk": risk, "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
         "positions": list(positions)})
    cycle(mode, broker, account, positions, risk)
    if once:
        return
    nxt = next_close(pd.Timestamp.now(tz="UTC"))
    while True:
        now = pd.Timestamp.now(tz="UTC")
        if now >= nxt + pd.Timedelta(seconds=20):  # small delay so the exchange has closed the bar
            try:
                cycle(mode, broker, account, positions, risk)
            except Exception as e:
                log({"event": "cycle_error", "error": repr(e)})
            nxt = next_close(now)
        elif mode == "paper" and positions:  # paper: emulate resting stop orders
            try:
                prices = {s: last_price(s) for s in positions}
                for s, px in broker.check_stops(positions, prices):
                    p = positions.pop(s)
                    log({"event": "stopped", "symbol": s, "exit": px, "entry": p.entry, "qty": p.qty,
                         "r": (px - p.entry) * p.qty / p.r_unit})
                    save_state(mode, account, positions)
            except Exception as e:
                log({"event": "poll_error", "error": repr(e)})
        time.sleep(poll)
