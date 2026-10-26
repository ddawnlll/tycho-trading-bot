"""Read the latest local paper report without connecting to any exchange."""
import json

import pandas as pd

from tycho.reporting import ROOT


def main():
    path = ROOT / "journal" / "nautilus" / "latest.json"
    if not path.exists():
        raise SystemExit("No paper account report yet (first snapshot arrives within one minute)")
    report = json.loads(path.read_text(encoding="utf-8"))
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(report["time"])).total_seconds()
    if age > 180 and report["status"] != "STOPPED":
        report["status"] = "STALE"
    print(json.dumps(dict(report, report_age_seconds=round(age, 1)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
