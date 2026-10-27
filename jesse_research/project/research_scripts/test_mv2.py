"""Synthetic self-checks for mv2_run (no market data). Run: ./run.sh research_scripts/test_mv2.py"""
import tempfile
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, "/work/research_scripts")

import mv2_run as R  # noqa: E402
from mv2_strategies import BAR_MS  # noqa: E402

T0 = 1_577_836_800_000  # 2020-01-01


def synth_1m(n, start, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.0008, n)))
    o = np.r_[c[0], c[:-1]]
    return np.c_[start + 60_000 * np.arange(n), o, c, np.maximum(o, c) * 1.0003, np.minimum(o, c) * 0.9997, np.ones(n)]


def test_all():
    tmp = Path(tempfile.mkdtemp())
    R.DATA, R.RES = tmp / "data", tmp / "res"
    a = synth_1m(60 * 24 * 40, T0 - 20 * 86_400_000)
    gapped = np.delete(a, np.s_[1000:1010], axis=0)
    (R.DATA / "1m" / "perp" / "XUSDT").mkdir(parents=True)
    import datetime as dt
    months = {}
    for row in gapped:
        months.setdefault(dt.datetime.utcfromtimestamp(row[0] / 1000).strftime("%Y-%m"), []).append(row)
    for mo, rows in months.items():
        np.save(R.DATA / "1m" / "perp" / "XUSDT" / f"{mo}.npy", np.array(rows))
    full, filled = R.load_1m("perp", "XUSDT", int(a[0, 0]), int(a[-1, 0]) + 60_000)
    assert filled == 10 and len(full) == len(a), (filled, len(full))
    assert np.all(np.diff(full[:, 0]) == 60_000)
    assert np.all(full[1000:1010, 1:5] == gapped[999, 2]) and np.all(full[1000:1010, 5] == 0)

    b = R.bars_4h(full)
    assert b[0, 0] % BAR_MS == 0 and np.isclose(b[0, 3], full[(full[:, 0] >= b[0, 0]) & (full[:, 0] < b[0, 0] + BAR_MS), 3].max())

    # funding: settlement at a bar close belongs to the bar starting there (open-at-settlement pays)
    (R.DATA / "funding" / "XUSDT").mkdir(parents=True)
    np.save(R.DATA / "funding" / "XUSDT" / "2020-01.npy", np.array([[T0 + 8 * 3_600_000 + 7, 0.001]]))
    ch = {"chunk": "2020-H1", "start_ms": T0, "end_ms": T0 + 10 * 86_400_000, "members": [{"symbol": "XUSDT", "slippage": 0.0}]}
    close_ts, f = R.bar_funding("perp", "XUSDT", b, ch)
    j = np.flatnonzero(f)
    assert len(j) == 1 and close_ts[j[0]] == T0 + 8 * 3_600_000, close_ts[j]

    # H3 book: one coin long, flat -> pays one entry and one exit cost, earns its return
    Rw, Fw = np.array([[0.10, 0.0], [0.0, 0.0]]), np.zeros((2, 2))
    sides = np.array([[1, 0], [0, 0]])
    book = R.h3_book(Rw, Fw, sides, np.array([0.001, 0.001]), k=1)
    assert np.allclose(book, [(0.10 - 0.001) / 2, -0.001 / 2]), book

    # Jesse run on the synthetic chunk with BuyHold: one trade, closed by stop_at, funding charged once
    m1, _ = R.load_1m("perp", "XUSDT", T0 - R.WARM * BAR_MS, ch["end_ms"])
    if m1[0, 0] <= T0 - R.WARM * BAR_MS:
        d = R.chunk_data("perp", ch, ch["members"][0])
    else:  # synthetic history shorter than warm-up: shrink warm-up for this check only
        R.WARM = int((T0 - m1[0, 0]) // BAR_MS)
        d = R.chunk_data("perp", ch, ch["members"][0])
    res = R.run_one("BH", {}, "base", "perp", ch, ch["members"][0], d)
    assert len(res["trades"]) == 1 and res["open_trades_at_end"] == 0, res["trades"]
    t = res["trades"][0]
    assert t["opened_at"] < T0 + 8 * 3_600_000 < t["closed_at"] <= ch["end_ms"]
    assert np.isclose(t["funding"], t["qty"] * full[full[:, 0] == T0 + 8 * 3_600_000, 1][0] * 0.001)
    assert np.isclose(res["equity"][-1][1], res["finishing_balance"] - t["funding"])
    # ML features must be finite outside a backtest (Jesse's default indicator window is 240 < 300)
    from mv2_strategies import features
    long4h = synth_1m(400, T0)
    long4h[:, 0] = T0 + BAR_MS * np.arange(400)
    assert np.isfinite(features(long4h)).all(), features(long4h)
    print("test_mv2: all checks passed")


if __name__ == "__main__":
    test_all()
