"""Jesse strategies for jesse_momentum_v2 (rules fixed in ../../preregistration.json).

Decisions run on closed 4h candles. Every strategy liquidates at `stop_at` (chunk end or the symbol's
last archived minute) so no position is left open when a chunk's backtest ends.
Optional class attributes, set by the runner before a backtest (the same process):
  MODEL     logistic-regression filter {mean, std, coef, intercept, thr} or None (H1/H2)
  SCHEDULE  (decision_ts array, side array) for H3, side in {+1, -1, 0}, forward-filled
"""
import bisect

import numpy as np
import jesse.indicators as ta
from jesse.strategies import Strategy

BAR_MS = 4 * 3_600_000


def features(c: np.ndarray) -> np.ndarray:
    """ML features at the close of the last candle of `c` (Jesse 4h rows: t, o, c, h, l, v)."""
    close, w = c[-1, 2], c[-330:]  # Jesse slices non-sequential indicators to warm_up_candles (330)
    return np.array([
        close / c[-181, 2] - 1,
        close / c[-43, 2] - 1,
        ta.atr(w, 14) / close,
        close / ta.ema(w, 300) - 1,
        c[-1, 5] / max(np.median(c[-31:-1, 5]), 1e-12),
    ])


def model_allows(model, c) -> bool:
    if model is None:
        return True
    x = (features(c) - model["mean"]) / model["std"]
    p = 1 / (1 + np.exp(-(x @ model["coef"] + model["intercept"])))
    return p >= model["thr"]


class _Base(Strategy):
    MODEL = None

    def hyperparameters(self):
        return [{"name": "stop_at", "type": float, "default": 4e12},
                {"name": "delay", "type": int, "default": 0}]

    @property
    def decision_ts(self):
        return self.current_candle[0] + BAR_MS

    @property
    def done(self):
        # a slot whose equity is <= 0 opens nothing new; its loss stays in the portfolio sum
        return self.decision_ts >= self.hp["stop_at"] or self.balance <= 0

    @property
    def sig(self):
        """Candles the signal is evaluated on: one bar old for the delayed-entry robustness test."""
        return self.candles[:-1] if self.hp["delay"] else self.candles

    def _stop_valid(self, mult):
        """No entry without a placeable protective stop (ATR so large that the stop would be <= 0)."""
        return self.price - mult * ta.atr(self.candles, 14) > 0

    def should_short(self):
        return False

    def should_cancel_entry(self):
        return True

    def go_short(self):
        pass


class H1Pullback(_Base):
    def hyperparameters(self):
        return super().hyperparameters() + [
            {"name": "rsi_entry", "type": float, "default": 45.0},
            {"name": "stop_atr", "type": float, "default": 2.0}]

    def should_long(self):
        c = self.sig
        if self.done or len(c) < 331:
            return False
        close = c[-1, 2]
        ema300 = ta.ema(c, 300)
        return (close > ema300 and ta.ema(c, 50) > ema300 and close / c[-181, 2] - 1 > 0
                and ta.rsi(c, 14) < self.hp["rsi_entry"] and self._stop_valid(self.hp["stop_atr"])
                and model_allows(self.MODEL, c))

    def go_long(self):
        stop = self.price - self.hp["stop_atr"] * ta.atr(self.candles, 14)
        qty = min(0.01 * self.balance / max(self.price - stop, 1e-12), self.balance / self.price)
        self.buy = qty, self.price
        self.stop_loss = qty, stop
        self.vars["bars"] = 0

    def update_position(self):
        self.vars["bars"] = self.vars.get("bars", 0) + 1
        c = self.candles
        if (self.done or ta.rsi(c, 14) > 70 or c[-1, 2] < ta.ema(c, 300) or self.vars["bars"] >= 60):
            self.liquidate()


class H2Breakout(_Base):
    def hyperparameters(self):
        return super().hyperparameters() + [
            {"name": "n", "type": int, "default": 30},
            {"name": "k", "type": float, "default": 3.0}]

    def should_long(self):
        c = self.sig
        n = self.hp["n"]
        if self.done or len(c) < 331:
            return False
        return (c[-1, 2] > c[-n - 1:-1, 3].max() and c[-1, 5] > 1.5 * np.median(c[-31:-1, 5])
                and c[-1, 2] > ta.ema(c, 300) and self._stop_valid(self.hp["k"]) and model_allows(self.MODEL, c))

    def go_long(self):
        stop = self.price - self.hp["k"] * ta.atr(self.candles, 14)
        qty = min(0.01 * self.balance / max(self.price - stop, 1e-12), self.balance / self.price)
        self.buy = qty, self.price
        self.stop_loss = qty, stop
        self.vars["stop"] = stop
        self.vars["hi"] = self.price

    def update_position(self):
        if self.done:
            self.liquidate()
            return
        self.vars["hi"] = max(self.vars["hi"], self.close)
        stop = max(self.vars["stop"], self.vars["hi"] - self.hp["k"] * ta.atr(self.candles, 14))
        if stop > self.vars["stop"]:
            self.vars["stop"] = stop
            self.stop_loss = self.position.qty, stop


class H3RelativeStrength(_Base):
    SCHEDULE = (np.zeros(0), np.zeros(0))

    def target(self):
        ts = self.decision_ts - (BAR_MS if self.hp["delay"] else 0)
        t, side = self.SCHEDULE
        i = bisect.bisect_right(t, ts) - 1
        return 0 if i < 0 or self.done else int(side[i])

    def should_long(self):
        return self.target() == 1

    def should_short(self):
        return self.target() == -1

    def go_long(self):
        self.buy = self.balance / self.price, self.price

    def go_short(self):
        self.sell = self.balance / self.price, self.price

    def update_position(self):
        side = 1 if self.is_long else -1
        if self.target() != side:
            self.liquidate()


class BuyHold(_Base):
    def should_long(self):
        return not self.done

    def go_long(self):
        self.buy = 0.99 * self.balance / self.price, self.price

    def update_position(self):
        if self.done:
            self.liquidate()


STRATEGIES = {"H1": H1Pullback, "H2": H2Breakout, "H3": H3RelativeStrength, "BH": BuyHold}
