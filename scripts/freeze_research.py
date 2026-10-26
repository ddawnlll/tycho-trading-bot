"""Freeze a new prospective window once; never overwrite an existing protocol."""
import json
from pathlib import Path

import pandas as pd

from tycho import config as C
from tycho.ensemble import configured_experts
from tycho.evidence import CANDIDATES, append_trial
from tycho.safety import expert_manifest, canonical_hash, source_hash
from tycho.locking import process_lock

ROOT = Path(__file__).resolve().parent.parent


def main():
    now = pd.Timestamp.now(tz="UTC")
    start = now.ceil(C.TIMEFRAME)
    experts = []
    for name in CANDIDATES:
        params, req = next(((p, r) for n, p, r in C.EXPERTS if n == name), ({}, {}))
        experts.append(configured_experts([(name, params, req)])[0])
    files = sorted(list((ROOT / "tycho").glob("*.py")) + [ROOT / "scripts" / "research_queue.py",
                                                         ROOT / "scripts" / "freeze_research.py"])
    protocol = dict(created=str(now), start=str(start), end=str(start + pd.Timedelta(days=180)),
                    status="PRE-EXPERIMENTAL / EVIDENCE-BOUND", economic_verdict="NO_ECONOMIC_CLAIM",
                    candidates=list(CANDIDATES), manifests={e.name: expert_manifest(e) for e in experts},
                    files={p.relative_to(ROOT).as_posix(): source_hash(p) for p in files},
                    policy=dict(min_days=180, min_closed_trades=30, max_drawdown=.20,
                                min_net_return=0, min_3x_cost_return=0, bootstrap_lower_quantile=.05,
                                bootstrap_block_days=30, null_max_p=.01, null_family_size=5,
                                seed=1729, repetitions=2000, owner_approval_required=True,
                                automatic_activation=False),
                    limitations=["Fixed 2026 universe has survivorship bias",
                                 "Hourly funding/open price proxy, not historical settlement mark tape",
                                 "Bar fills do not prove exchange fill quality",
                                 "30-day bootstrap blocks may be short for long holding periods",
                                 "AWAITING_OWNER_APPROVAL is a review queue, never an economic claim"])
    path = ROOT / "research" / "protocol.json"
    path.parent.mkdir(exist_ok=True)
    with process_lock(ROOT / "live_state" / "research.lock"):
        with path.open("x", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)
        append_trial(ROOT / "research" / "trials.jsonl", dict(event="preregistered", protocol_hash=canonical_hash(protocol),
                                                               start=protocol["start"], end=protocol["end"]))
    print(f"Frozen {path}; future observations only; no strategy activation")


if __name__ == "__main__":
    main()
