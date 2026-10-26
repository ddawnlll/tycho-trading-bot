import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tycho import config as C
from tycho.ensemble import configured_experts
from tycho.evidence import append_trial, read_trials, review, cost_adjusted_equity
from tycho.locking import process_lock
from tycho.reporting import PaperReport
from tycho.safety import paper_only, validate_bars, expert_manifest, source_hash
from tycho.nautilus_live import build_node
from tycho.live import BinanceBroker, main


@pytest.mark.parametrize("mode,risk", [("live", .005), ("testnet", .005), ("typo", .005),
                                       ("paper", 0), ("paper", -.1), ("paper", float("nan")),
                                       ("paper", float("inf")), ("paper", .03)])
def test_node_refuses_exchange_execution_before_build(mode, risk):
    with pytest.raises(ValueError):
        build_node(mode, risk)


def test_legacy_exchange_cannot_be_instantiated():
    with pytest.raises(RuntimeError, match="disabled"):
        BinanceBroker(False)
    with pytest.raises(RuntimeError, match="retired"):
        main("live")


def test_candidate_cannot_be_added_to_paper_by_config(monkeypatch):
    monkeypatch.setattr(C, "EXPERTS", C.EXPERTS + [("range_revert", {}, {})])
    with pytest.raises(ValueError, match="authorized only"):
        paper_only("paper", .005)


def test_manifest_tracks_parameter_mutation():
    expert = configured_experts()[0]
    before = expert_manifest(expert)
    expert.params["entry_n"] = 999
    assert before != expert_manifest(expert)


def test_source_hash_survives_git_newline_conversion(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\nprint(x)\n")
    b.write_bytes(b"x = 1\r\nprint(x)\r\n")
    assert source_hash(a) == source_hash(b)


def frames(monkeypatch):
    monkeypatch.setattr(C, "UNIVERSE", ["BTCUSDT", "ETHUSDT"])
    monkeypatch.setattr(C, "REGIME_WARMUP_BARS", 5)
    monkeypatch.setattr(C, "WARMUP_BARS", 5)
    idx = pd.date_range("2026-01-01", periods=5, freq="4h", tz="UTC")
    df = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1.), index=idx)
    return {s: df.copy() for s in C.UNIVERSE}, idx[-1]


@pytest.mark.parametrize("bad", ["missing", "future", "gap", "nan", "ohlc", "warmup", "duplicate"])
def test_bad_snapshot_abstains(monkeypatch, bad):
    bars, last = frames(monkeypatch)
    validate_bars(bars, last)
    if bad == "missing":
        bars.pop("ETHUSDT")
    elif bad == "future":
        bars["ETHUSDT"].index += pd.Timedelta("4h")
    elif bad == "gap":
        bars["ETHUSDT"].index = bars["ETHUSDT"].index.where(
            bars["ETHUSDT"].index != bars["ETHUSDT"].index[1], bars["ETHUSDT"].index[1] - pd.Timedelta("1h"))
    elif bad == "nan":
        bars["ETHUSDT"].iloc[-1, 0] = np.nan
    elif bad == "ohlc":
        bars["ETHUSDT"].iloc[-1, 1] = 95
    elif bad == "warmup":
        bars["ETHUSDT"] = bars["ETHUSDT"].iloc[1:]
    else:
        bars["ETHUSDT"].index = pd.DatetimeIndex([bars["ETHUSDT"].index[0]] * 4 + [last])
    with pytest.raises(ValueError):
        validate_bars(bars, last)


def test_report_costs_funding_and_drawdown(tmp_path):
    report = PaperReport(tmp_path)
    event = SimpleNamespace(trade_id="fill-1", instrument_id=SimpleNamespace(symbol=SimpleNamespace(value="BTCUSDT-PERP")),
                            last_qty=2, last_px=100, order_side="BUY", client_order_id="o1", commission=".1 USDT", ts_event=100)
    report.fill(event)
    report.fill(event)
    assert report.slippage == pytest.approx(.04)
    rows = [dict(fundingTime=1, markPrice="100", fundingRate="0.001")]
    report.settle_funding("BTCUSDT", rows)
    report.settle_funding("BTCUSDT", rows)
    assert report.funding == pytest.approx(.2)
    first = report.snapshot(10000, [])
    report.snapshot(9000, [dict(symbol="BTCUSDT")])
    last = report.snapshot(9800, [])
    assert first["pnl"] == pytest.approx(-.24)
    assert last["max_drawdown"] > last["drawdown"]
    assert json.loads((tmp_path / "latest.json").read_text())["pnl"] == pytest.approx(-200.24)


def test_trial_tamper_detection(tmp_path):
    path = tmp_path / "trials.jsonl"
    append_trial(path, dict(expert="a", result="FAIL"))
    append_trial(path, dict(expert="b", result="BLOCKED"))
    assert len(read_trials(path)) == 2
    path.write_text(path.read_text().replace("FAIL", "PASS"))
    with pytest.raises(ValueError, match="integrity"):
        append_trial(path, dict(expert="c"))


def test_gate_never_promotes_old_missing_or_adverse_evidence():
    kw = dict(fresh=True, days=200, trades=50, net_return=.05, stress_return=.02,
              max_drawdown=.10, bootstrap_lower=.001, null_p=.005, paper_days=35, paper_trades=30)
    assert review(**kw)["verdict"] == "AWAITING_OWNER_APPROVAL"
    assert review(**dict(kw, fresh=False))["verdict"] == "BLOCKED"
    assert review(**dict(kw, stress_return=-.01))["verdict"] == "FAIL"
    assert review(**dict(kw, null_p=None))["verdict"] == "BLOCKED"
    assert review(**dict(kw, null_p=float("nan")))["verdict"] == "INVALID"


def test_single_process_lock(tmp_path):
    with process_lock(tmp_path / "paper.lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            with process_lock(tmp_path / "paper.lock"):
                pass


def test_cost_overlay_known_inventory_and_stress():
    idx = pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC")
    result = dict(equity=pd.Series([10000., 10000., 10000.], index=idx),
                  fills=pd.DataFrame([dict(instrument_id="BTCUSDT-PERP.BINANCE", filled_qty=2,
                                           avg_px=100, ts_last=idx[0])]),
                  positions=pd.DataFrame([dict(instrument_id="BTCUSDT-PERP.BINANCE", ts_opened=idx[0],
                                                ts_closed=pd.NaT, peak_qty=2, entry="BUY")]))
    hourly = {"BTCUSDT": pd.DataFrame(dict(open=[100., 100., 100.], funding=[0., .001, 0.]), index=idx)}
    base, costs = cost_adjusted_equity(result, hourly)
    stress, _ = cost_adjusted_equity(result, hourly, 3)
    assert costs["slippage"] == pytest.approx(.04)
    assert costs["funding"] == pytest.approx(.2)
    assert base.iloc[-1] == pytest.approx(9999.76)
    assert stress.iloc[-1] == pytest.approx(9999.08)


def test_live_strategy_fills_and_reports_on_native_simulated_exchange(tmp_path, monkeypatch):
    from tycho import nautilus_bt as NB, nautilus_live as NL
    from tests.test_nautilus_orders import _data, _sig

    # Mechanics test always runs, including CI without cached Binance history.
    monkeypatch.setattr(NB, "exchange_filters", lambda: {"BTCUSDT": {"tick": "0.1", "step": "0.001"}})
    monkeypatch.setattr(NL, "STATE_DIR", tmp_path)
    monkeypatch.setattr(NL, "PaperReport", lambda: PaperReport(tmp_path))
    original = NB.EnsembleStrategy

    class PaperHarness(NL.EnsembleLiveStrategy):
        def __init__(self, config, plan, instruments, bar_types, expected):
            super().__init__(config, "paper")
            self.plan, self.inst, self.bar_types = plan, instruments, bar_types
            self.expected, self.cols = expected, list(plan)

        def on_start(self):
            original.on_start(self)

        def on_bar(self, bar):
            original.on_bar(self, bar)

    monkeypatch.setattr(NB, "EnsembleStrategy", PaperHarness)
    df = _data([100, 102, 104, 106, 108, 110, 108, 100, 90, 85])
    key = "trend_long:BTCUSDT"
    result = NB.run({"BTCUSDT": df}, {key: df}, {key: _sig(len(df), 1, 95)}, df.index[0], df.index[-1])
    assert len(result["positions"]) == 1
    assert not result["positions"].ts_closed.isna().any()
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["status"] == "STOPPED"
    assert latest["pnl"] < 0
    assert latest["slippage_reserve"] > 0
    events = [json.loads(line) for p in tmp_path.glob("*.jsonl") for line in p.read_text().splitlines()]
    fills = [e for e in events if e["event"] == "fill"]
    # Nautilus may split a market order across several native fills.
    assert len(fills) >= 2
    assert sum(e["quantity"] * (1 if e["side"] == "1" else -1) for e in fills) == pytest.approx(0)
    assert sum(e["quantity"] for e in fills if e["side"] == "1") == pytest.approx(
        float(result["positions"].peak_qty.iloc[0]))
    assert latest["positions"] == []


def test_pending_entries_reserve_portfolio_gross():
    from tycho.nautilus_bt import EnsembleStrategy
    fake = SimpleNamespace(inst={"BTCUSDT": SimpleNamespace(id="BTC")},
                           held={"BTCUSDT": {"pending_qty": 10}}, close={"BTCUSDT": 100},
                           portfolio=SimpleNamespace(net_position=lambda _: 2))
    assert EnsembleStrategy._gross(fake) == 1000


@pytest.mark.parametrize("condition", ["ready", "pause", "stale_stream"])
def test_fresh_bar_tick_decides_once_and_respects_pause_and_staleness(tmp_path, monkeypatch, condition):
    import asyncio
    from tycho import nautilus_live as NL

    bars, last = frames(monkeypatch)
    now = last + pd.Timedelta("4h30s")
    monkeypatch.setattr(NL, "STATE_DIR", tmp_path)
    monkeypatch.setattr(NL, "server_now", lambda: now)
    monkeypatch.setattr(NL, "fetch_all_bars", lambda: bars)
    monkeypatch.setattr(NL, "last_price", lambda _: 100.)
    frame = pd.DataFrame(dict(long_entry=[True], long_exit=[False], long_stop=[90.], long_trail=[np.nan]), index=[last.value])
    monkeypatch.setattr(NL, "latest_plan", lambda _: ({"trend_long:BTCUSDT": frame.copy()}, {"btc_bull": True}))
    if condition == "pause":
        (tmp_path / "PAUSE").touch()
    calls = []
    fake = SimpleNamespace(done_bar=None, report=SimpleNamespace(inventory_events=[], event=lambda *a, **k: None),
                           _check_risk=lambda: None, MAX_DECISION_DELAY=pd.Timedelta("30min"),
                           manifests={e.name: expert_manifest(e) for e in configured_experts()},
                           last_mark_ns={s: now.value if condition != "stale_stream" else 0 for s in C.UNIVERSE},
                           close={}, halted=False, held={}, _persist=lambda: None,
                           log=SimpleNamespace(info=lambda _: None, warning=lambda _: None))
    fake._decide = lambda t: calls.append((t, bool(fake.plan["trend_long:BTCUSDT"].long_entry.iloc[0])))
    if condition == "stale_stream":
        with pytest.raises(ValueError, match="missing/stale"):
            asyncio.run(NL.EnsembleLiveStrategy._tick_inner(fake))
        assert calls == []
    else:
        asyncio.run(NL.EnsembleLiveStrategy._tick_inner(fake))
        asyncio.run(NL.EnsembleLiveStrategy._tick_inner(fake))
        assert calls == [(last.value, condition == "ready")]
