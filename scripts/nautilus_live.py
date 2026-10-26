"""Run the expert ensemble on a NautilusTrader TradingNode.

  python -m scripts.nautilus_live                    # sandbox: live Binance data, simulated fills (no keys)
  python -m scripts.nautilus_live --mode testnet     # Binance DEMO futures account (demo API keys in env)
  python -m scripts.nautilus_live --mode live --i-understand-real-money
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.nautilus_live import build_node

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sandbox", "testnet", "live"], default="sandbox")
    ap.add_argument("--risk", type=float, default=C.RISK_DEFAULT)
    ap.add_argument("--i-understand-real-money", action="store_true")
    a = ap.parse_args()
    if a.mode == "live" and not a.i_understand_real_money:
        sys.exit("refusing live mode without --i-understand-real-money")
    if a.risk > 0.02:
        sys.exit("risk > 2% per trade is outside anything this strategy was validated for")
    node = build_node(a.mode, a.risk)
    try:
        node.run()
    finally:
        node.dispose()
