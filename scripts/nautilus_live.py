"""Run the expert ensemble on a NautilusTrader TradingNode.

  python -m scripts.nautilus_live                    # sandbox: live Binance data, simulated fills (no keys)
  python -m scripts.nautilus_live --run-seconds 155  # bounded smoke session
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C
from tycho.nautilus_live import build_node
from tycho.locking import process_lock
from tycho.live import STATE_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "sandbox"], default="paper")
    ap.add_argument("--risk", type=float, default=C.RISK_DEFAULT)
    ap.add_argument("--run-seconds", type=int, help="stop cleanly after a bounded smoke session")
    a = ap.parse_args()
    with process_lock(STATE_DIR / "paper.lock"):
        node = build_node(a.mode, a.risk)
        if a.run_seconds is not None:
            if a.run_seconds <= 0:
                ap.error("--run-seconds must be positive")
            node.get_event_loop().call_later(a.run_seconds, node.stop)
        try:
            node.run()
        finally:
            node.dispose()
