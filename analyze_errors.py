"""Phase 0 error analysis: show per-field disagreements from the saved run,
so we can see whether misses are real extraction errors or scoring noise.

Run:  python analyze_errors.py
"""
import json

from distill.schema import TARGET_FIELDS
from distill.score import _match


def main(path: str = "outputs/phase0_results.json"):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]

    for f in TARGET_FIELDS:
        misses = [(r["pred"].get(f), r["gold"].get(f)) for r in rows
                  if not _match(f, r["pred"].get(f), r["gold"].get(f))]
        print(f"\n===== {f}: {len(misses)} mismatches / {len(rows)} =====")
        for pred, gold in misses:
            if gold is None:
                tag = "FALSE-POSITIVE"
            elif pred is None:
                tag = "MISSED"
            else:
                tag = "WRONG"
            print(f"  [{tag:14s}] pred={pred!r}  gold={gold!r}")


if __name__ == "__main__":
    main()
