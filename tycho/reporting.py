"""Paper session reports; Nautilus owns fills, orders, balances and positions.

Costs absent from the sandbox (slippage and funding) are disclosed as conservative
reporting deductions. They never masquerade as exchange fills or profitability evidence.
"""
import json
import math
from pathlib import Path
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


class PaperReport:
    def __init__(self, directory=None):
        self.session = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        self.directory = Path(directory or ROOT / "journal" / "nautilus")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{self.session}.jsonl"
        self.initial = 10000.0
        self.peak = self.initial
        self.max_dd = 0.0
        self.slippage = 0.0
        self.funding = 0.0
        self.fills = set()
        self.inventory_events = []
        self.funding_events = set()
        self.started_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        self.event("session_start", execution="NAUTILUS_SANDBOX", real_money=False,
                   note="Fresh simulated account each process start; prior sessions are retained, never resumed",
                   economic_verdict="NO_ECONOMIC_CLAIM")

    def event(self, kind, **fields):
        row = clean(dict(time=str(pd.Timestamp.now(tz="UTC")), session=self.session, event=kind, **fields))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False, default=str) + "\n")
        return row

    def fill(self, event):
        trade = str(event.trade_id)
        if trade in self.fills:
            return
        self.fills.add(trade)
        symbol = event.instrument_id.symbol.value.replace("-PERP", "")
        bps = 2 if symbol in {"BTCUSDT", "ETHUSDT"} else 5
        debit = float(event.last_qty) * float(event.last_px) * bps / 10000
        self.slippage += debit
        side = 1 if str(event.order_side) in {"BUY", "1"} else -1
        self.inventory_events.append((symbol, int(event.ts_event), side * float(event.last_qty)))
        self.event("fill", instrument=str(event.instrument_id), order=str(event.client_order_id),
                   trade=trade, side=str(event.order_side), quantity=float(event.last_qty),
                   price=float(event.last_px), commission=str(event.commission), slippage_reserve=debit)

    def settle_funding(self, symbol, rows):
        for row in rows:
            stamp = int(row["fundingTime"])
            key = (symbol, stamp)
            if key in self.funding_events:
                continue
            qty = sum(q for s, t, q in self.inventory_events if s == symbol and t < stamp * 1_000_000)
            mark, rate = float(row["markPrice"]), float(row["fundingRate"])
            if not math.isfinite(mark) or mark <= 0 or not math.isfinite(rate):
                raise ValueError("invalid funding settlement data")
            debit = qty * mark * rate
            self.funding += debit
            self.funding_events.add(key)
            self.event("funding", symbol=symbol, funding_time=stamp, quantity=qty, mark=mark, rate=rate, debit=debit)

    def snapshot(self, engine_equity, positions, **fields):
        net = engine_equity - self.slippage - self.funding
        self.peak = max(self.peak, net)
        dd = max(0.0, 1 - net / self.peak)
        self.max_dd = max(self.max_dd, dd)
        row = self.event("account", engine_equity=engine_equity, equity_after_cost_reserves=net,
                         pnl=net - self.initial, drawdown=dd, max_drawdown=self.max_dd,
                         slippage_reserve=self.slippage, funding_reserve=self.funding,
                         positions=positions, economic_verdict="NO_ECONOMIC_CLAIM", **fields)
        tmp = self.directory / "latest.tmp"
        tmp.write_text(json.dumps(row, indent=2, allow_nan=False, default=str), encoding="utf-8")
        tmp.replace(self.directory / "latest.json")
        return row
