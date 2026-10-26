"""Run the strategy. Default is PAPER (no keys, no real orders).

  python -m scripts.live                       # paper, loops forever
  python -m scripts.live --once                # one paper decision cycle
  python -m scripts.live --mode testnet        # Binance demo account (needs demo API keys in env)
  python -m scripts.live --mode live --i-understand-real-money   # REAL MONEY
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.live import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "testnet", "live"], default="paper")
    ap.add_argument("--risk", type=float, default=C.RISK_DEFAULT, help="risk per trade as share of equity")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--i-understand-real-money", action="store_true")
    a = ap.parse_args()
    if a.mode == "live" and not a.i_understand_real_money:
        sys.exit("refusing live mode without --i-understand-real-money")
    if a.risk > 0.02:
        sys.exit("risk > 2% per trade is outside anything this strategy was validated for")
    main(a.mode, a.risk, a.once)
