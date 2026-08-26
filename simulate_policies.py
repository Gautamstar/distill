"""Simulate blame policies for the arithmetic check and price the tradeoff.

When `subtotal + tax + tip = total` fails, one of those fields is wrong - but
the current rule caps all of them, and measurement says 24 of 32 capped fields
were actually correct. Freeing the collateral raises auto-accept coverage and
costs some accuracy on the accepted slice. This prices that exchange instead of
guessing at it.

Uncapped fields are assumed to return to auto-accept: only 4% of wrongly-capped
receipts showed any sampling disagreement, so their pre-cap confidence was
almost always unanimous. That makes these coverage figures a slight
over-estimate and the accuracy figures a slight under-estimate.

Run:  python simulate_policies.py
"""
import json

PRICES = ("subtotal_price", "tax_price", "tip_price", "total_price")
CAP = 0.5
ACCEPT = 0.999


def load(path="outputs/confidence_results_v3.json"):
    return json.load(open(path, encoding="utf-8"))["rows"]


def simulate(rows, keep_capped):
    """keep_capped: which fields stay capped when the arithmetic check fires."""
    n = ok = accepted = accepted_ok = 0
    for r in rows:
        for name, d in r["fields"].items():
            n += 1
            ok += bool(d["correct"])
            capped = name in PRICES and abs(d["confidence"] - CAP) < 1e-6
            if capped and name not in keep_capped:
                conf = 1.0          # returns to auto-accept
            else:
                conf = d["confidence"]
            if conf >= ACCEPT:
                accepted += 1
                accepted_ok += bool(d["correct"])
    return {
        "coverage": accepted / n,
        "accept_accuracy": accepted_ok / accepted if accepted else 0,
        "accepted": accepted,
        "n": n,
        # what the pipeline actually delivers: accepted as-is, flagged corrected
        "delivered": (accepted_ok + (n - accepted)) / n,
    }


POLICIES = {
    "current (cap every price field)": set(PRICES),
    "cap subtotal + total": {"subtotal_price", "total_price"},
    "cap subtotal only": {"subtotal_price"},
    "cap nothing (no arithmetic gate)": set(),
}


def main():
    rows = load()
    print("Auto-accept policy when the arithmetic check fires\n")
    print(f"  {'policy':34s} {'coverage':>9s} {'accept acc':>11s} {'delivered':>10s}")
    for label, keep in POLICIES.items():
        s = simulate(rows, keep)
        print(f"  {label:34s} {s['coverage']:8.1%} {s['accept_accuracy']:11.1%} "
              f"{s['delivered']:10.1%}")
    print("\n  coverage    = share of fields a human never touches")
    print("  accept acc  = accuracy of what ships unreviewed")
    print("  delivered   = end-to-end accuracy, assuming review fixes what it sees")


if __name__ == "__main__":
    main()
