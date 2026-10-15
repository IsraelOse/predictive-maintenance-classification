"""
Reproducibility check.

Fingerprints every analytical output in the project and compares it against a
stored reference, so that re-running the twelve stages can be shown to produce
identical results rather than merely claimed to.

Wall-clock fields are excluded, because they are the one thing that legitimately
varies between runs: `fit_seconds` and `search_seconds` in tuning.json, and the
`mean_fit_time` column of the four cv_results tables. Nothing else is allowed
to move.

    python src/reproducibility_check.py            compare against the reference
    python src/reproducibility_check.py --write    record a new reference

Exit code 0 means every analytical value matches.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config as C

REFERENCE = C.STORE / "_fingerprint.json"
TIMING_KEY = re.compile(r"seconds|fit_time", re.I)
TIMING_COL = re.compile(r"fit_time|score_time", re.I)


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items()
                if not TIMING_KEY.search(str(k))}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def digest_store():
    out = {}
    for path in sorted(C.STORE.glob("*.json")):
        if path.name.startswith("_"):
            continue
        data = strip_timing(json.loads(path.read_text(encoding="utf-8")))
        blob = json.dumps(data, sort_keys=True, default=str).encode()
        out[path.name] = hashlib.sha256(blob).hexdigest()[:16]
    return out


def digest_tables():
    out = {}
    for path in sorted(C.TABLES.glob("*.csv")):
        df = pd.read_csv(path)
        keep = [c for c in df.columns if not TIMING_COL.search(str(c))]
        if path.name == "tbl27_selected_configurations.csv":
            keep = [c for c in keep if c != "search_seconds"]
        if path.name in ("tbl25_default_model_cv.csv", "tbl26_tuned_model_cv.csv"):
            keep = [c for c in keep if c != "fit_seconds"]
        blob = df[keep].to_csv(index=False).encode()
        out[path.name] = hashlib.sha256(blob).hexdigest()[:16]
    return out


def main():
    write = "--write" in sys.argv
    current = {
        "seed": C.SEED,
        "store": digest_store(),
        "tables": digest_tables(),
    }
    overall = hashlib.sha256(
        json.dumps(current, sort_keys=True).encode()).hexdigest()[:16]
    current["overall"] = overall

    print("=" * 70)
    print("REPRODUCIBILITY CHECK  (seed %d)" % C.SEED)
    print("=" * 70)
    print("%d stored results, %d tables fingerprinted."
          % (len(current["store"]), len(current["tables"])))
    print("Excluded from the fingerprint: fit_seconds, search_seconds and the "
          "mean_fit_time column, which are wall-clock and vary by machine.")
    print("\nOverall fingerprint: %s" % overall)

    if write or not REFERENCE.exists():
        REFERENCE.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print("\nReference written to %s" % REFERENCE)
        print("Re-run the twelve stages and call this script again to verify.")
        return 0

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if ref.get("overall") == overall:
        print("\nMATCHES the recorded reference. Every analytical value in this "
              "project reproduces exactly at seed %d." % C.SEED)
        return 0

    print("\nDIFFERS from the recorded reference. Changed items:")
    n = 0
    for group in ("store", "tables"):
        for name, h in current[group].items():
            old = ref.get(group, {}).get(name)
            if old != h:
                n += 1
                print("  %-8s %-46s %s -> %s"
                      % (group, name, old or "(new)", h))
    for group in ("store", "tables"):
        for name in ref.get(group, {}):
            if name not in current[group]:
                n += 1
                print("  %-8s %-46s removed" % (group, name))
    print("\n%d differences. If this is an intended change, re-record with "
          "--write." % n)
    return 1


if __name__ == "__main__":
    sys.exit(main())
