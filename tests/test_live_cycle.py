"""cycle() safety with a fake broker: restart idempotency, never-unprotected stops, stale bars."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tycho import live
from tycho.live import Decision, Position

NOW = pd.Timestamp("2026-10-26 12:00:30", tz="UTC")
LAST = pd.Timestamp("2026-10-26 08:00", tz="UTC")


class FakeBroker:
    def __init__(self, fail_stop=False):
        self.buys, self.sells, self.stops, self.fail_stop = [], [], [], fail_stop

    def equity(self, positions, prices):
        return 10_000.0

    def buy(self, s, qty, px):
        self.buys.append(s)
        return px, qty

    def sell(self, p, px):
        self.sells.append(p.symbol)
        return px

    def place_stop(self, p, stop):
        if self.fail_stop:
            raise RuntimeError("-2021 order would immediately trigger")
        self.stops.append((p.symbol, stop))
        return f"id{len(self.stops)}"

    def check_stops(self, positions, prices):
        return []


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(live, "JOURNAL_DIR", tmp_path / "journal")
    monkeypatch.setattr(live, "server_now", lambda: NOW)
    bars = pd.DataFrame({"close": [1.0]}, index=[LAST])
    monkeypatch.setattr(live, "fetch_all_bars", lambda: {"BTCUSDT": bars})
    monkeypatch.setattr(live, "last_price", lambda s: 100.0)
    holder = {}

    def fake_decide(b, positions):
        return holder["dec"]
    monkeypatch.setattr(live, "decide", fake_decide)
    return holder


def test_restart_does_not_rebuy_same_bar(env):
    env["dec"] = Decision(bar_time=LAST + live.BAR, entries=[("trend_long", "ETHUSDT", 90.0)], regime_bull=True)
    b, acct, pos = FakeBroker(), {}, {}
    assert live.cycle("paper", b, acct, pos, 0.005)
    # simulate a crash + restart: reload persisted state, run the same bar again
    acct2, pos2 = live.load_state("paper")
    assert "ETHUSDT" in pos2 and acct2["last_bar"] == str(LAST + live.BAR)
    assert live.cycle("paper", b, acct2, pos2, 0.005)
    assert b.buys == ["ETHUSDT"]


def test_entry_flattened_when_stop_cannot_be_placed(env):
    env["dec"] = Decision(bar_time=LAST + live.BAR, entries=[("trend_long", "ETHUSDT", 90.0)], regime_bull=True)
    b, pos = FakeBroker(fail_stop=True), {}
    live.cycle("paper", b, {}, pos, 0.005)
    assert b.buys == ["ETHUSDT"] and b.sells == ["ETHUSDT"] and pos == {}


def test_failed_stop_move_keeps_old_stop_and_retries(env):
    pos = {"ETHUSDT": Position("ETHUSDT", 1.0, 100.0, 90.0, 10.0, "x", stop_order_id="old")}
    env["dec"] = Decision(bar_time=LAST + live.BAR, stop_moves={"ETHUSDT": 95.0}, regime_bull=True)
    live.cycle("paper", FakeBroker(fail_stop=True), {}, pos, 0.005)
    assert pos["ETHUSDT"].stop == 90.0 and pos["ETHUSDT"].stop_order_id == "old"
    env["dec"] = Decision(bar_time=LAST + 2 * live.BAR, stop_moves={"ETHUSDT": 95.0}, regime_bull=True)
    live.cycle("paper", FakeBroker(), {}, pos, 0.005)
    assert pos["ETHUSDT"].stop == 95.0 and pos["ETHUSDT"].stop_order_id == "id1"


def test_stale_bars_are_not_traded(env, monkeypatch):
    monkeypatch.setattr(live, "fetch_all_bars",
                        lambda: {"BTCUSDT": pd.DataFrame({"close": [1.0]}, index=[LAST - live.BAR])})
    env["dec"] = Decision(bar_time=LAST, entries=[("trend_long", "ETHUSDT", 90.0)], regime_bull=True)
    b = FakeBroker()
    assert live.cycle("paper", b, {}, {}, 0.005) is False and b.buys == []


def test_signal_exit_and_one_expert_per_coin(env):
    pos = {"ETHUSDT": Position("ETHUSDT", 1.0, 100.0, 90.0, 10.0, "x", stop_order_id="s", expert="xs_momentum")}
    env["dec"] = Decision(bar_time=LAST + live.BAR, exits=["ETHUSDT"],
                          entries=[("trend_long", "SOLUSDT", 90.0), ("xs_momentum", "SOLUSDT", 80.0)], regime_bull=True)
    b = FakeBroker()
    live.cycle("paper", b, {}, pos, 0.005)
    assert b.sells == ["ETHUSDT"] and "ETHUSDT" not in pos
    assert b.buys == ["SOLUSDT"] and pos["SOLUSDT"].expert == "trend_long" and pos["SOLUSDT"].stop == 90.0


def test_failed_exit_keeps_position_tracked(env):
    class SellFails(FakeBroker):
        def sell(self, p, px):
            raise RuntimeError("network")
    pos = {"ETHUSDT": Position("ETHUSDT", 1.0, 100.0, 90.0, 10.0, "x", stop_order_id="s", expert="xs_momentum")}
    env["dec"] = Decision(bar_time=LAST + live.BAR, exits=["ETHUSDT"], regime_bull=True)
    with pytest.raises(RuntimeError):
        live.cycle("paper", SellFails(), {}, pos, 0.005)
    assert "ETHUSDT" in pos
