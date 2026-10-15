"""
Stage 00 - Environment capture

Writes requirements.txt with the exact versions this analysis ran on, and
session_info.txt with the interpreter, platform and full package list. Run it
last, so that the recorded environment is the one that actually produced the
results, but it is numbered 00 because it describes the environment the other
stages assume.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C

# The packages this project imports directly. Everything else in the venv is a
# transitive dependency and is recorded in session_info.txt rather than pinned.
DIRECT = [
    "pandas", "numpy", "scikit-learn", "xgboost", "imbalanced-learn",
    "matplotlib", "seaborn", "scipy", "ucimlrepo", "python-docx", "openpyxl",
    "nbformat", "nbconvert", "jupyter", "ipykernel", "pywin32",
]


def frozen():
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def main():
    pkgs = frozen()
    by_name = {}
    for line in pkgs:
        if "==" in line:
            name, ver = line.split("==", 1)
            by_name[name.lower().replace("_", "-")] = (name, ver)

    req = C.ROOT / "requirements.txt"
    lines = [
        "# DAMO510 Take-Home Assignment 2",
        "# Direct dependencies, pinned to the versions this analysis ran on.",
        "# Reproduce with:  python -m venv .venv && .venv/Scripts/pip install -r requirements.txt",
        "",
    ]
    missing = []
    for p in DIRECT:
        key = p.lower()
        if key in by_name:
            name, ver = by_name[key]
            lines.append("%s==%s" % (name, ver))
        else:
            missing.append(p)
    req.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("requirements.txt written with %d pinned packages" % (len(DIRECT) - len(missing)))
    if missing:
        print("  not installed, omitted: %s" % ", ".join(missing))

    info = C.ROOT / "session_info.txt"
    with open(info, "w", encoding="utf-8") as fh:
        fh.write("DAMO510 Take-Home Assignment 2 - session information\n")
        fh.write("=" * 62 + "\n\n")
        fh.write("Generated      : %s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        fh.write("Assigned seed  : %d\n" % C.SEED)
        fh.write("Python         : %s\n" % sys.version.replace("\n", " "))
        fh.write("Executable     : %s\n" % sys.executable)
        fh.write("Platform       : %s\n" % platform.platform())
        fh.write("Processor      : %s\n" % (platform.processor() or "unknown"))
        fh.write("Machine        : %s\n" % platform.machine())
        fh.write("\nProject constants\n")
        fh.write("-" * 62 + "\n")
        fh.write("SEED             = %d\n" % C.SEED)
        fh.write("TEST_SIZE        = %.2f\n" % C.TEST_SIZE)
        fh.write("CV_FOLDS         = %d\n" % C.CV_FOLDS)
        fh.write("SELECTION_METRIC = %s\n" % C.SELECTION_METRIC)
        fh.write("COST_FN / COST_FP= %.0f / %.0f  (ratio %.0f:1)\n"
                 % (C.COST_FN, C.COST_FP, C.COST_RATIO))
        fh.write("ROBUSTNESS_SEEDS = %s\n" % C.ROBUSTNESS_SEEDS)
        fh.write("FIG_DPI          = %d\n" % C.FIG_DPI)
        fh.write("\nDirect dependencies\n")
        fh.write("-" * 62 + "\n")
        for p in DIRECT:
            key = p.lower()
            if key in by_name:
                name, ver = by_name[key]
                fh.write("%-22s %s\n" % (name, ver))
        fh.write("\nComplete environment (pip freeze, %d packages)\n" % len(pkgs))
        fh.write("-" * 62 + "\n")
        for line in pkgs:
            fh.write(line + "\n")
    print("session_info.txt written (%d packages recorded)" % len(pkgs))

    print("\n--- requirements.txt ---")
    print(req.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
