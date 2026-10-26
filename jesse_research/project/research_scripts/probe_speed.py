"""Probe Jesse fast-mode speed/memory on one year of synthetic 1m candles (no market data)."""
import resource
import time

import numpy as np
import jesse.helpers as jh
import jesse.indicators as ta
from jesse import utils
from jesse.research import backtest
from jesse.strategies import Strategy


class Probe(Strategy):
    def should_long(self):
        return self.close > ta.donchian(self.candles[:-1], 20).upperband

    def go_long(self):
        self.buy = utils.size_to_qty(self.balance * 0.5, self.price), self.price
        self.stop_loss = self.position.qty if False else None

    def update_position(self):
        if self.close < ta.ema(self.candles, 50):
            self.liquidate()

    def should_cancel_entry(self):
        return True


def candles(n, start=1609459200000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * 1.0005
    lo = np.minimum(open_, close) * 0.9995
    t = start + 60_000 * np.arange(n)
    return np.c_[t, open_, close, hi, lo, np.full(n, 10.0)].astype(float)


if __name__ == '__main__':
    ex, sym = 'Binance Perpetual Futures', 'BTC-USDT'
    warm = candles(240 * 210, start=1609459200000 - 240 * 210 * 60_000, seed=1)
    trade = candles(525_600)
    cfg = {'starting_balance': 10_000, 'fee': 0.0007, 'type': 'futures', 'futures_leverage': 2,
           'futures_leverage_mode': 'cross', 'exchange': ex, 'warm_up_candles': 210}
    routes = [{'exchange': ex, 'strategy': Probe, 'symbol': sym, 'timeframe': '4h'}]
    t0 = time.time()
    r = backtest(cfg, routes, [], {jh.key(ex, sym): {'exchange': ex, 'symbol': sym, 'candles': trade}},
                 {jh.key(ex, sym): {'exchange': ex, 'symbol': sym, 'candles': warm}},
                 fast_mode=True, generate_equity_curve=True)
    print('seconds', round(time.time() - t0, 1), 'maxrss_MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    print('keys', sorted(r), 'trades', r['metrics'].get('total'))
    if r.get('trades'):
        print('trade0', r['trades'][0])
    eq = r.get('equity_curve')
    print('equity type', type(eq), (eq[0].keys() if isinstance(eq, list) and eq else None))
    if isinstance(eq, list) and eq:
        print({k: (v if k != 'data' else v[:3]) for k, v in eq[0].items()}, len(eq[0].get('data', [])))
    print({k: r['metrics'][k] for k in list(r['metrics'])[:40]})
