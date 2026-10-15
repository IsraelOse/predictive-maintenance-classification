"""Ask Word itself how many pages the body occupies.

Opens the generated DOCX through COM, repaginates, and reports the page on
which each numbered heading starts. This is the authoritative check on the
5-to-6-page body requirement; the word-count estimate in verify_report.py is
only a proxy.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

import win32com.client as win32

WD_ACTIVE_END_PAGE = 3

def main():
    path = str(C.REPORT / "DAMO510_A2_Report.docx")
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(path, ReadOnly=True)
    try:
        doc.Repaginate()
        total = doc.ComputeStatistics(2)   # wdStatisticPages
        marks = {}
        for p in doc.Paragraphs:
            t = p.Range.Text.strip()
            for key in ["1. Introduction", "2. Data audit", "3. Exploratory",
                        "4. Preprocessing", "5. Modelling", "6. Results",
                        "7. Threshold", "8. Interpretation",
                        "9. Final recommendation", "10. Limitations",
                        "11. References", "12. Analytical Decision Record",
                        "Appendix A", "Appendix B"]:
                if t.startswith(key) and key not in marks:
                    marks[key] = p.Range.Information(WD_ACTIVE_END_PAGE)
        print("Total pages in document: %d" % total)
        print("\nSection start pages:")
        for k in ["1. Introduction", "2. Data audit", "3. Exploratory",
                  "4. Preprocessing", "5. Modelling", "6. Results",
                  "7. Threshold", "8. Interpretation", "9. Final recommendation",
                  "10. Limitations", "11. References",
                  "12. Analytical Decision Record", "Appendix A", "Appendix B"]:
            if k in marks:
                print("  %-32s page %d" % (k, marks[k]))
        body_start = marks.get("1. Introduction")
        body_end = marks.get("11. References")
        if body_start and body_end:
            pages = body_end - body_start
            print("\nBODY (sections 1 to 10): pages %d to %d = %d pages"
                  % (body_start, body_end - 1, pages))
            print("Requirement is 5 to 6 pages: %s"
                  % ("PASS" if 5 <= pages <= 6 else "FAIL"))
    finally:
        doc.Close(False)
        word.Quit()

if __name__ == "__main__":
    main()
