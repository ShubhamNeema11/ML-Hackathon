"""Automatic go/no-go decisions of aws/run_pipeline.sh. Prints one word.

  python aws/gates.py pilot   -> go | no      (aws/compare_ce.py wrote logs/pilot_gate.json)
  python aws/gates.py b4      -> b4 | b2      (B4 must beat B2 by > 0.0005 on the half no threshold was tuned on)
  python aws/gates.py pass2   -> pass2 | pass1 (pass 2 must beat pass 1 by > 0.0005 on the report half, same procedure)
"""
import json
import os
import sys
from pathlib import Path

R = Path(os.environ.get("ER_ROOT", "."))
MARGIN = 0.0005


def load(p):
    return json.loads((R / p).read_text()) if (R / p).exists() else None


def main():
    what = sys.argv[1]
    if what == "pilot":
        g = load("logs/pilot_gate.json")
        print("go" if g and g.get("go") else "no")
    elif what == "b4":
        a, b = load("models/decision_b2.json"), load("models/decision_b4.json")
        ok = a and b and b["held"]["early-stop half"]["official"] > a["held"]["early-stop half"]["official"] + MARGIN
        print("b4" if ok else "b2")
    elif what == "pass2":
        d = load("models/decision_pass2.json")
        ok = d and d["held"]["report-half"]["official"] > d["pass1_held"]["report-half"]["official"] + MARGIN
        print("pass2" if ok else "pass1")


if __name__ == "__main__":
    main()
