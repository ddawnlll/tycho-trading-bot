"""Download untouched holdout data from data.binance.vision (public archives, no keys).

1) Every Binance USDT-M perpetual that ever had monthly 4h klines, INCLUDING delisted ones
   (survivorship-bias control), minus the 12-coin dev universe. Klines + funding.
2) Spot 4h klines 2017-08..2019-12 for the dev-universe coins that existed (pre-perp era, 2018 bear).

Hard cutoff: nothing after 2026-09 (monthly archives), so the frozen prospective window
(2026-10-26 20:00 UTC onward, research/protocol.json) is never read.

Usage: python -m scripts.download_holdout
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import config as C

BASE = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
OUT = Path(__file__).resolve().parent.parent / "data" / "holdout"
LAST_MONTH = "2026-09"          # frozen window starts 2026-10-26; never fetch later months
PERP_FIRST = "2020-01"
SPOT = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT", "DOGEUSDT"]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qvol", "n", "tbv", "tbq", "x"]
S = requests.Session()


def s3_list(prefix: str) -> list[str]:
    out, marker = [], ""
    while True:
        r = S.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=60)
        r.raise_for_status()
        out += re.findall(r"<Prefix>([^<]+)</Prefix>", r.text)[1:] + re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", r.text)
        if "<IsTruncated>true</IsTruncated>" not in r.text or not m:
            return out
        marker = m.group(1)


def fetch_csv(url: str, names=None) -> pd.DataFrame | None:
    for _ in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            raw = z.read(z.namelist()[0]).decode()
            first = raw.split("\n", 1)[0]
            has_header = not first[:1].isdigit()
            return pd.read_csv(io.StringIO(raw), header=0 if has_header else None, names=None if has_header else names)
        except Exception:
            continue
    return None


def klines(market: str, sym: str, months: list[str]) -> pd.DataFrame:
    frames = []
    for m in months:
        df = fetch_csv(f"{BASE}/data/{market}/monthly/klines/{sym}/4h/{sym}-4h-{m}.zip", COLS)
        if df is not None and len(df):
            df.columns = COLS[:len(df.columns)]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    t = df["open_time"].astype("int64")
    t = t.where(t < 10**14, t // 1000)  # 2025+ spot archives use microseconds
    out = df[["open", "high", "low", "close", "volume", "qvol"]].astype(float)
    out.index = pd.to_datetime(t, unit="ms", utc=True)
    out.index.name = "time"
    return out[~out.index.duplicated()].sort_index()


def funding(sym: str, months: list[str]) -> pd.Series:
    frames = []
    for m in months:
        df = fetch_csv(f"{BASE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames)
    t = pd.to_datetime(df["calc_time"].astype("int64"), unit="ms", utc=True).dt.floor("4h")
    s = pd.Series(df["last_funding_rate"].astype(float).to_numpy(), index=t)
    return s.groupby(level=0).sum()  # summed per 4h bar, like tycho.data.resample


def months_of(keys: list[str]) -> list[str]:
    ms = sorted({m.group(1) for k in keys if (m := re.search(r"-(\d{4}-\d{2})\.zip$", k))})
    return [m for m in ms if m <= LAST_MONTH]


def perp_job(sym: str):
    path = OUT / "perp" / f"{sym}_4h.parquet"
    if path.exists():
        return sym, "cached"
    months = [m for m in months_of(s3_list(f"data/futures/um/monthly/klines/{sym}/4h/")) if m >= PERP_FIRST]
    if len(months) < 12:
        return sym, "skip<12m"
    k = klines("futures/um", sym, months)
    if k.empty:
        return sym, "empty"
    f = funding(sym, months)
    k["funding"] = f.reindex(k.index).fillna(0.0)
    k = k[k.index < pd.Timestamp("2026-10-01", tz="UTC")]
    path.parent.mkdir(parents=True, exist_ok=True)
    k.to_parquet(path)
    return sym, f"{len(k)} bars {k.index[0].date()}..{k.index[-1].date()}"


def spot_job(sym: str):
    path = OUT / "spot" / f"{sym}_4h.parquet"
    if path.exists():
        return sym, "cached"
    months = [m for m in months_of(s3_list(f"data/spot/monthly/klines/{sym}/4h/")) if m <= "2019-12"]
    k = klines("spot", sym, months).assign(funding=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    k.to_parquet(path)
    return sym, f"{len(k)} bars {k.index[0].date()}..{k.index[-1].date()}"


def main():
    syms = [p.rstrip("/").split("/")[-1] for p in s3_list("data/futures/um/monthly/klines/")]
    perps = sorted(s for s in syms if s.endswith("USDT") and s not in C.UNIVERSE)
    print(f"{len(syms)} archived UM symbols, {len(perps)} USDT perps outside the dev universe", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for sym, msg in ex.map(spot_job, SPOT):
            print("spot", sym, msg, flush=True)
    done = 0
    with ThreadPoolExecutor(16) as ex:
        for sym, msg in ex.map(perp_job, perps):
            done += 1
            if done % 25 == 0 or "bars" not in msg:
                print(f"[{done}/{len(perps)}] {sym} {msg}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
