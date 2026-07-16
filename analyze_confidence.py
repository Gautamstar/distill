"""Deep-dive on the confidence run: calibration table and the auto-accept curve.

Reads outputs/confidence_results.json (written by run_confidence.py) and answers:
  1. Calibration: at each confidence level, how accurate are those fields?
     (Does 0.6 confidence actually mean ~60% correct?)
  2. Policy curve: for each threshold, if we auto-accept fields at or above it,
     what share of fields do we automate and how accurate is that slice?
  3. Per-field: which fields earn high confidence and which stay uncertain.

Run:  python analyze_confidence.py
"""
import json
from collections import defaultdict

from distill.schema import TARGET_FIELDS


def main(path: str = "outputs/confidence_results.json"):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]

    # Flatten to (field, confidence, correct) triples.
    flat = [(f, r["fields"][f]["confidence"], r["fields"][f]["correct"])
            for r in rows for f in TARGET_FIELDS]
    n = len(flat)

    print(f"{len(rows)} receipts, {n} scored fields\n")

    print("== Calibration: accuracy at each confidence level ==")
    by_conf = defaultdict(list)
    for _, conf, ok in flat:
        by_conf[round(conf, 2)].append(ok)
    for conf in sorted(by_conf, reverse=True):
        oks = by_conf[conf]
        print(f"  conf={conf:.2f}   n={len(oks):4d}   accuracy={sum(oks) / len(oks):.3f}")

    print("\n== Auto-accept policy curve (accept fields with conf >= t) ==")
    print(f"  {'threshold':>9}  {'coverage':>8}  {'accepted-acc':>12}  {'flagged-acc':>11}")
    for t in (1.0, 0.8, 0.6, 0.4, 0.2):
        acc_set = [ok for _, c, ok in flat if c >= t - 1e-9]
        rej_set = [ok for _, c, ok in flat if c < t - 1e-9]
        cov = len(acc_set) / n if n else 0.0
        a1 = sum(acc_set) / len(acc_set) if acc_set else float("nan")
        a0 = sum(rej_set) / len(rej_set) if rej_set else float("nan")
        print(f"  {t:9.1f}  {cov:8.1%}  {a1:12.3f}  {a0:11.3f}")

    print("\n== Per-field: mean confidence and accuracy of the confident slice ==")
    for f in TARGET_FIELDS:
        triples = [(c, ok) for ff, c, ok in flat if ff == f]
        confident = [ok for c, ok in triples if c >= 0.999]
        mean_c = sum(c for c, _ in triples) / len(triples)
        cov = len(confident) / len(triples)
        acc = sum(confident) / len(confident) if confident else float("nan")
        print(f"  {f:16s} mean_conf={mean_c:.2f}  conf1.0-share={cov:.1%}  "
              f"conf1.0-accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
