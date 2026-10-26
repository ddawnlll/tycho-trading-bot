"""Usage: python scripts/download.py [SYMBOL ...]   (default: full universe)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho.data import load

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
           "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"]
if __name__ == "__main__":
    for s in (sys.argv[1:] or SYMBOLS):
        df = load(s, refresh=True)
        print(s, len(df), df.index[0], df.index[-1], "funding events:", int((df.funding != 0).sum()), flush=True)
