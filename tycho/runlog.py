"""Persist every research/backtest run under runs/<timestamp>_<tag>/ and append to runs/index.csv."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent.parent / "runs"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RUNS.parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


def new_run(tag: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = RUNS / f"{ts}_{tag}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def save(d: Path, config: dict, summary: dict, tables: dict[str, pd.DataFrame] | None = None,
         notes: str = "") -> None:
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for name, t in (tables or {}).items():
        t.to_csv(d / f"{name}.csv")
    if notes:
        (d / "NOTES.md").write_text(notes, encoding="utf-8")
    idx = RUNS / "index.csv"
    new = not idx.exists()
    with idx.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "git", "headline"])
        w.writerow([d.name, _git_rev(), summary.get("headline", "")])
