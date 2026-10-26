"""Download + validate public Binance archives for jesse_momentum_v2 (no keys, no exchange API).

Stages (each resumable; finished files are skipped):
  1. list USDT-M perpetual and spot USDT symbols from the data.binance.vision S3 listing (delisted included)
  2. daily klines -> data/daily/{market}/{SYM}.npy  (universe ranking only)
  3. universe per half-year chunk -> data/universe.json (rules in preregistration.json)
  4. 1m klines (+ perp funding) for chunk members incl. warm-up -> data/1m/{market}/{SYM}/{YYYY-MM}.npy
  5. data/manifest.json: sources, sha256, row counts, missing minutes, OHLC anomalies, delistings

Hard cutoff: nothing after 2026-09 is requested (Tycho frozen window starts 2026-10-26 20:00 UTC).
Usage (inside the Jesse container): python fetch_data.py /data
"""
import calendar
import datetime as dt
import hashlib
import io
import json
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

BASE = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
LAST_MONTH = "2026-09"
MARKETS = {  # market -> (archive path, first month, last month, first chunk, last chunk)
    "perp": ("futures/um", "2019-09", LAST_MONTH, (2020, 1), (2026, 2)),
    "spot": ("spot", "2017-08", "2019-12", (2018, 1), (2019, 2)),
}
TYCHO_HOLDOUT_B = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT", "DOGEUSDT"}
STABLE = {"USDC", "BUSD", "TUSD", "USDP", "FDUSD", "DAI", "EUR", "AEUR", "USDS", "PAX", "UST", "USTC", "USDE", "USDSB"}
INDEX = {"BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD"}
TOP_N, HIST_DAYS, RANK_DAYS, WARMUP_DAYS = 15, 60, 60, 56  # warm-up 330 x 4h = 55 days (+1 spare)
S = requests.Session()
LOCK = threading.Lock()


def get(url, **kw):
    for attempt in range(6):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def s3_list(prefix):
    out, marker = [], ""
    while True:
        r = get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        out += re.findall(r"<Prefix>([^<]+)</Prefix>", r.text)[1:] + re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", r.text)
        if "<IsTruncated>true</IsTruncated>" not in r.text or not m:
            return out
        marker = m.group(1)


def tradable(sym, market):
    if not sym.endswith("USDT") or "_" in sym:
        return False
    base = sym[:-4]
    if base in STABLE or base in INDEX:
        return False
    if market == "spot" and (base.endswith(("UP", "DOWN", "BULL", "BEAR")) and len(base) > 4):
        return False
    return True


def fetch_zip(url):
    """Return (csv text, sha256) after verifying the archive's published CHECKSUM, or None if absent."""
    r = get(url)
    if r is None:
        return None
    digest = hashlib.sha256(r.content).hexdigest()
    c = get(url + ".CHECKSUM")
    if c is None or c.text.split()[0] != digest:
        raise ValueError(f"checksum mismatch or missing for {url}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return z.read(z.namelist()[0]).decode(), digest


def parse_klines(text):
    rows = [ln.split(",") for ln in text.strip().split("\n") if ln and ln[0].isdigit()]
    a = np.array([[float(x) for x in r[:8]] for r in rows]) if rows else np.zeros((0, 8))
    if len(a):
        a[:, 0] = np.where(a[:, 0] > 1e14, a[:, 0] // 1000, a[:, 0])  # microsecond archives
    return a  # open_time, open, high, low, close, volume, close_time, quote_volume


def months(first, last):
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def ms(d):
    return int(calendar.timegm(d.timetuple()) * 1000)


def chunks(first, last):
    (y, h), out = first, []
    while (y, h) <= last:
        start = dt.datetime(y, 1 if h == 1 else 7, 1)
        end = dt.datetime(y, 7, 1) if h == 1 else dt.datetime(y + 1, 1, 1)
        out.append((f"{y}-H{h}", start, end))
        y, h = (y, 2) if h == 1 else (y + 1, 1)
    return out


def daily_job(args):
    market, sym, root, manifest = args
    path = root / "daily" / market / f"{sym}.npy"
    if path.exists():
        return sym, "cached"
    arch, first, last = MARKETS[market][:3]
    keys = s3_list(f"data/{arch}/monthly/klines/{sym}/1d/")
    have = sorted({m.group(1) for k in keys if (m := re.search(r"-(\d{4}-\d{2})\.zip$", k))})
    parts = []
    for mo in [m for m in have if first <= m <= last]:
        got = fetch_zip(f"{BASE}/data/{arch}/monthly/klines/{sym}/1d/{sym}-1d-{mo}.zip")
        if got:
            parts.append(parse_klines(got[0]))
            with LOCK:
                manifest.setdefault(f"daily/{market}/{sym}", []).append([mo, got[1]])
    a = np.concatenate(parts) if parts else np.zeros((0, 8))
    a = a[np.unique(a[:, 0], return_index=True)[1]] if len(a) else a
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)
    return sym, f"{len(a)} days"


def universe(root, market):
    first, last = MARKETS[market][3:]
    daily = {p.stem: np.load(p) for p in sorted((root / "daily" / market).glob("*.npy"))}
    out = []
    for name, start, end in chunks(first, last):
        t0 = ms(start)
        cand = []
        for sym, a in daily.items():
            if market == "spot" and sym in TYCHO_HOLDOUT_B:
                continue
            if not len(a) or a[0, 0] > t0 - HIST_DAYS * 86_400_000 or not (a[:, 0] == t0 - 86_400_000).any():
                continue
            w = a[(a[:, 0] >= t0 - RANK_DAYS * 86_400_000) & (a[:, 0] < t0)]
            cand.append((float(np.median(w[:, 7])), sym))
        cand.sort(reverse=True)
        top = [{"symbol": s, "median_qvol": v,
                "slippage": 0.0002 if v >= 1e9 else 0.0005 if v >= 2e8 else 0.001,
                "last_daily_ms": int(daily[s][-1, 0])} for v, s in cand[:TOP_N]]
        out.append({"chunk": name, "start_ms": t0, "end_ms": ms(end), "eligible": len(cand), "members": top})
    return out


def minute_job(args):
    market, sym, mo, root, manifest = args
    path = root / "1m" / market / sym / f"{mo}.npy"
    if path.exists():
        return
    arch = MARKETS[market][0]
    got = fetch_zip(f"{BASE}/data/{arch}/monthly/klines/{sym}/1m/{sym}-1m-{mo}.zip")
    if got is None:
        with LOCK:
            manifest[f"1m/{market}/{sym}/{mo}"] = {"missing_file": True}
        return
    k = parse_klines(got[0])
    k = k[np.unique(k[:, 0], return_index=True)[1]]
    y, m = map(int, mo.split("-"))
    m0, m1 = ms(dt.datetime(y, m, 1)), ms(dt.datetime(y + (m == 12), m % 12 + 1, 1))
    k = k[(k[:, 0] >= m0) & (k[:, 0] < m1)]
    bad = int(((k[:, 2] < np.maximum(k[:, 1], k[:, 4])) | (k[:, 3] > np.minimum(k[:, 1], k[:, 4]))
               | (k[:, 1:5] <= 0).any(1) | (k[:, 5] < 0)).sum())
    first, last = (int(k[0, 0]), int(k[-1, 0])) if len(k) else (None, None)
    # missing minutes inside the listed life of this month (before first / after last row = not listed)
    inside = (last - first) // 60_000 + 1 if len(k) else 0
    with LOCK:
        manifest[f"1m/{market}/{sym}/{mo}"] = {"sha256": got[1], "rows": len(k), "first_ms": first, "last_ms": last,
                                               "missing_inside": int(inside - len(k)), "ohlc_anomalies": bad}
    jesse = np.c_[k[:, 0], k[:, 1], k[:, 4], k[:, 2], k[:, 3], k[:, 5]]  # Jesse order: t, o, c, h, l, v
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, jesse)


def funding_job(args):
    sym, mo, root, manifest = args
    path = root / "funding" / sym / f"{mo}.npy"
    if path.exists():
        return
    got = fetch_zip(f"{BASE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{mo}.zip")
    rows = []
    if got:
        for ln in got[0].strip().split("\n"):
            p = ln.split(",")
            if p[0].isdigit():
                rows.append([float(p[0]), float(p[-1])])  # calc_time, last_funding_rate
        entry = {"sha256": got[1], "rows": len(rows)}
    else:
        entry = {"missing_file": True}
    with LOCK:
        manifest[f"funding/{sym}/{mo}"] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows).reshape(-1, 2))


def main(root):
    root = Path(root)
    mpath = root / "manifest_raw.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    listing = {}
    for market, (arch, *_ ) in MARKETS.items():
        syms = [p.rstrip("/").split("/")[-1] for p in s3_list(f"data/{arch}/monthly/klines/")]
        listing[market] = sorted(s for s in syms if tradable(s, market))
        print(market, len(syms), "archived symbols,", len(listing[market]), "kept", flush=True)
    for market in MARKETS:
        with ThreadPoolExecutor(24) as ex:
            for i, (sym, msg) in enumerate(ex.map(daily_job, [(market, s, root, manifest) for s in listing[market]])):
                if i % 100 == 0:
                    print("daily", market, i, sym, msg, flush=True)
        mpath.write_text(json.dumps(manifest))
    uni = {m: universe(root, m) for m in MARKETS}
    (root / "universe.json").write_text(json.dumps(uni, indent=1))
    jobs, fjobs = set(), set()
    for market, (_, first, last, *_ ) in MARKETS.items():
        for ch in uni[market]:
            w0 = dt.datetime.utcfromtimestamp((ch["start_ms"] - WARMUP_DAYS * 86_400_000) / 1000).strftime("%Y-%m")
            e = dt.datetime.utcfromtimestamp((ch["end_ms"] - 1) / 1000).strftime("%Y-%m")
            for mem in ch["members"]:
                for mo in months(max(w0, first), min(e, last)):
                    jobs.add((market, mem["symbol"], mo))
                    if market == "perp":
                        fjobs.add((mem["symbol"], mo))
    print(len(jobs), "1m symbol-months,", len(fjobs), "funding symbol-months", flush=True)
    with ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(minute_job, [(*j, root, manifest) for j in sorted(jobs)])):
            if i % 100 == 0:
                print("1m", i, flush=True)
                with LOCK:
                    mpath.write_text(json.dumps(manifest))
        list(ex.map(funding_job, [(*j, root, manifest) for j in sorted(fjobs)]))
    mpath.write_text(json.dumps(manifest))
    delisted = {m: [s for s in listing[m] if (root / "daily" / m / f"{s}.npy").exists()
                    and len(a := np.load(root / "daily" / m / f"{s}.npy"))
                    and a[-1, 0] < ms(dt.datetime(*(2026, 9, 29) if m == "perp" else (2019, 12, 30)))] for m in MARKETS}
    summary = {
        "generated_utc": dt.datetime.utcnow().isoformat(),
        "source": BASE, "hard_cutoff_month": LAST_MONTH,
        "symbols_kept": {m: len(v) for m, v in listing.items()},
        "ended_before_period_end": {m: len(v) for m, v in delisted.items()},
        "minute_files": sum(1 for k, v in manifest.items() if k.startswith("1m/") and "sha256" in v),
        "minute_files_missing": sorted(k for k, v in manifest.items() if k.startswith("1m/") and v.get("missing_file")),
        "missing_minutes_total": sum(v.get("missing_inside", 0) for k, v in manifest.items() if k.startswith("1m/")),
        "missing_minutes_by_file": {k: v["missing_inside"] for k, v in manifest.items()
                                    if k.startswith("1m/") and v.get("missing_inside", 0) > 0},
        "ohlc_anomalies_total": sum(v.get("ohlc_anomalies", 0) for k, v in manifest.items() if k.startswith("1m/")),
        "funding_files_missing": sorted(k for k, v in manifest.items() if k.startswith("funding/") and v.get("missing_file")),
        "members_ending_inside_chunk": [(m, ch["chunk"], mem["symbol"]) for m in MARKETS for ch in uni[m]
                                        for mem in ch["members"] if mem["last_daily_ms"] < ch["end_ms"] - 86_400_000],
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_minutes_by_file"}, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
