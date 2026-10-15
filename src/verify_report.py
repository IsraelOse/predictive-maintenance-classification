"""
Voice, length and cross-reference check on the generated DOCX.

Reads report/DAMO510_A2_Report.docx back and reports:
  word count of the body (sections 1 to 10) and an estimated page count
  any em-dash or en-dash
  every banned phrase from the writing brief
  appendix references that point at nothing, and figures the body never cites

src/page_count.py is the authoritative page check; the estimate here is a
quick proxy calibrated against it.

Run after build_report.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document

import config as C

BANNED = [
    "not only", "it's worth noting", "it is worth noting", "delve", "leverage",
    "crucial", "pivotal", "landscape", "underscores", "showcases",
    "in today's world", "this analysis demonstrates that", "furthermore",
    "moreover", "in conclusion", "firstly", "secondly", "seamless",
    "cutting-edge", "game-changer", "tapestry", "realm", "testament",
    "navigate the", "dive into", "unlock", "harness",
]
BANNED_REGEX = [
    (r"\bnot\s+[a-z ]{1,25},\s*but\b", "the 'not X, but Y' construction"),
    (r"\brobust(?!ness)\b", "'robust' as filler"),
]

DOC = C.REPORT / "DAMO510_A2_Report.docx"


def refs(text, letter):
    """Every appendix reference in the text, expanding plural forms and
    'A15 to A17' ranges. The body legitimately writes 'Figures A15 to A17',
    and a naive 'Figure (A\\d+)' pattern silently misses both the plural
    and the range, which is how this check produced false alarms before."""
    found = set(int(m) for m in re.findall(r"\b%s(\d+)\b" % letter, text))
    for lo, hi in re.findall(r"\b%s(\d+)\s+to\s+%s(\d+)\b" % (letter, letter),
                             text):
        found.update(range(int(lo), int(hi) + 1))
    return found


def main():
    doc = Document(DOC)
    paras = [p.text for p in doc.paragraphs]

    start = next(i for i, t in enumerate(paras) if t.startswith("1. Introduction"))
    end = next(i for i, t in enumerate(paras) if t.startswith("11. References"))
    body = [t for t in paras[start:end] if t.strip()]
    text = "\n".join(body)
    words = len(text.split())

    print("=" * 70)
    print("REPORT VERIFICATION")
    print("=" * 70)
    print("File: %s" % DOC)
    print("Body word count (sections 1 to 10): %d" % words)
    # Calibrated against Word's own pagination for this layout.
    print("Estimated pages at ~600 words per page: %.1f" % (words / 600.0))
    ok_len = 2900 <= words <= 3700
    print("Length verdict: %s  (page_count.py is authoritative)"
          % ("WITHIN RANGE" if ok_len else "OUT OF RANGE"))

    full = "\n".join(paras)
    print("\n--- Dash check (whole document) ---")
    em, en = full.count("—"), full.count("–")
    print("Em-dashes: %d" % em)
    print("En-dashes: %d" % en)

    print("\n--- Banned phrase check (body only) ---")
    low = text.lower()
    hits = []
    for b in BANNED:
        n = low.count(b)
        if n:
            hits.append((b, n))
    for rx, label in BANNED_REGEX:
        found = re.findall(rx, low)
        if found:
            hits.append((label, len(found)))
    if hits:
        for h, n in hits:
            print("  FOUND %-45s x%d" % (h, n))
    else:
        print("  none found")

    print("\n--- Cross-reference check ---")
    n_fig = sum(1 for p in paras if p.startswith("Figure A"))
    n_tab = sum(1 for p in paras if p.startswith("Table B"))
    print("Figure captions in Appendix A: %d" % n_fig)
    print("Table captions in Appendix B : %d" % n_tab)

    bad_f = sorted(i for i in refs(full, "A") if i > n_fig)
    bad_t = sorted(i for i in refs(full, "B") if i > n_tab)
    print("Dangling figure references: %s" % (bad_f or "none"))
    print("Dangling table references : %s" % (bad_t or "none"))

    cited_f, cited_t = refs(text, "A"), refs(text, "B")
    print("Figures cited in the body: %d of %d" % (len(cited_f), n_fig))
    print("Tables cited in the body : %d of %d" % (len(cited_t), n_tab))
    uncited_f = ["A%d" % i for i in range(1, n_fig + 1) if i not in cited_f]
    uncited_t = ["B%d" % i for i in range(1, n_tab + 1) if i not in cited_t]
    print("Figures never cited: %s" % (uncited_f or "none"))
    print("Tables never cited : %s" % (uncited_t or "none"))

    print("\n--- Vague appendix pointers ---")
    vague = [p for p in body if re.search(r"see appendix(?!\s+[AB])", p, re.I)]
    print("Instances of 'see appendix' without a specific reference: %d" % len(vague))

    print("\n--- Seed statement ---")
    print("Seed 93 appears %d times in the document." % full.count("93"))

    # Every figure must be cited: the appendix rule requires the body to carry
    # the finding and point at the picture. Tables are indexed evidence, so a
    # handful of unreferenced supporting tables is acceptable and is reported
    # rather than failed.
    ok = (em == 0 and en == 0 and not hits and not bad_f and not bad_t
          and ok_len and not uncited_f and not vague)
    print("\n" + "=" * 70)
    print("OVERALL: %s" % ("CLEAN" if ok else "ISSUES ABOVE NEED ATTENTION"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
