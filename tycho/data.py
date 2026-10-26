"""Download and cache Binance USDT-M perpetual klines and funding rates (public REST, no keys)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

FAPI = "https://fapi.binance.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MIN_GAP = 0.3  # seconds between requests: 1500-limit kline = 10 weight -> ~2000 weight/min max


_last = [0.0]


def _get(path: str, params: dict) -> list:
    for attempt in range(6):
        wait = _MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        r = requests.get(FAPI + path, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 418:
            raise RuntimeError("Binance IP ban (418); stop and wait")
        time.sleep(int(r.headers.get("Retry-After", 2 ** attempt)))
    r.raise_for_status()
    return []


def fetch_klines(symbol: str, interval: str = "1h", start: str = "2020-01-01") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list = []
    while True:
        batch = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                         "startTime": start_ms, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < 1500:
            break
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qvol", "n", "tbv", "tbq", "x"])
    df = df[["open_time", "open", "high", "low", "close", "volume", "qvol"]].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop(columns="open_time").drop_duplicates("time").set_index("time").sort_index()


def fetch_funding(symbol: str, start: str = "2020-01-01") -> pd.Series:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows: list = []
    while True:
        batch = _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
    if not rows:
        return pd.Series(dtype=float, name="funding")
    df = pd.DataFrame(rows)
    s = pd.Series(df["fundingRate"].astype(float).values,
                  index=pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.floor("h"),
                  name="funding")
    return s[~s.index.duplicated()].sort_index()


def load(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """1h bars with a 'funding' column (rate charged at that hour, 0 otherwise)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{symbol}_1h.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    k = fetch_klines(symbol)
    f = fetch_funding(symbol)
    k["funding"] = f.reindex(k.index).fillna(0.0)
    k = k[k.index < pd.Timestamp.now(tz="UTC").floor("h")]  # drop still-forming bar
    k.to_parquet(path)
    return k


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "qvol": "sum", "funding": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])
