"""Why arithmetic validation over-flags, and what signal separates the cases.

Calibration showed fields capped at 0.5 by the arithmetic check are 70.6%
correct: the check is routing work to humans that humans did not need to see.
This looks at the capped fields and asks whether anything already available
distinguishes the ones that were wrong from the ones that were fine.

Run:  python analyze_overflagging.py
"""
import json
from collections import Counter

from distill.score import _to_float

PRICES = ("subtotal_price", "tax_price", "tip_price", "total_price")


def main(path="outputs/confidence_results_v3.json"):
    rows = json.load(open(path, encoding="utf-8"))["rows"]

    capped = []
    for r in rows:
        f = r["fields"]
        for name in PRICES:
            d = f.get(name)
            # validation_flag is not persisted in saved runs, but with k=5 the
            # sampling agreement can only be .2/.4/.6/.8/1.0 - exactly 0.5 can
            # only have come from the validation cap.
            if d and abs(d["confidence"] - 0.5) < 1e-6:
                # raw sampling agreement, before validation capped it
                capped.append({"field": name, "correct": bool(d["correct"]),
                               "receipt": f})

    ok = [c for c in capped if c["correct"]]
    bad = [c for c in capped if not c["correct"]]
    print(f"arithmetic-capped fields: {len(capped)}  "
          f"({len(ok)} correct, {len(bad)} wrong)\n")

    print("Which fields get capped, and how often that was a mistake:")
    by_field = Counter(c["field"] for c in capped)
    by_field_ok = Counter(c["field"] for c in ok)
    for name, n in by_field.most_common():
        print(f"  {name:16s} capped {n:3d}   wrongly {by_field_ok[name]:3d} "
              f"({by_field_ok[name] / n:.0%})")

    # Hypothesis: when the equation fails, the field that is actually wrong is
    # usually the one the samples already disagreed about. Capping every price
    # field ignores that the model often flagged the culprit itself.
    print("\nDid the receipt contain a field the samples disagreed on?")
    for label, group in (("wrongly capped (was correct)", ok),
                         ("rightly capped (was wrong)", bad)):
        had_dissent = 0
        for c in group:
            confs = [c["receipt"][p]["confidence"] for p in PRICES
                     if c["receipt"].get(p) and c["receipt"][p]["value"] is not None]
            # 0.5 is the cap itself, so look for genuine sampling disagreement
            if any(x < 0.999 and abs(x - 0.5) > 1e-6 for x in confs):
                had_dissent += 1
        pct = had_dissent / len(group) if group else 0
        print(f"  {label:30s} {had_dissent:3d}/{len(group):3d}  ({pct:.0%})")

    print("\nHow far off is the arithmetic when it fails?")
    for label, group in (("wrongly capped", ok), ("rightly capped", bad)):
        gaps = []
        for c in group:
            f = c["receipt"]
            g = lambda k: (_to_float(f[k]["value"])
                           if f.get(k) and f[k]["value"] else None)
            sub, tax, tip, tot = g("subtotal_price"), g("tax_price"), g("tip_price"), g("total_price")
            if sub is not None and tot is not None:
                gaps.append(abs(sub + (tax or 0) + (tip or 0) - tot))
        if gaps:
            gaps.sort()
            print(f"  {label:16s} n={len(gaps):3d}  median {gaps[len(gaps) // 2]:8.2f}  "
                  f"min {gaps[0]:.2f}  max {gaps[-1]:.2f}")


if __name__ == "__main__":
    main()
