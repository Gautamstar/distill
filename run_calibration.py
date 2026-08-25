"""Report calibration of the confidence signal, and what recalibrating buys.

Run:  python run_calibration.py [outputs/confidence_results_v3.json]
"""
import json
import sys

from distill.calibrate import bins, ece, mce, fit, apply_fit


def main(path="outputs/confidence_results_v3.json"):
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    b = bins(rows)
    n = sum(x["n"] for x in b)

    print(f"{n} field decisions from {len(rows)} receipts\n")
    print("== Reliability: does the number mean what it says? ==")
    print(f"  {'stated':>7} {'n':>5} {'observed':>9} {'gap':>8}")
    for x in b:
        flag = "  <-- overconfident" if x["gap"] < -0.1 else (
               "  <-- underconfident" if x["gap"] > 0.1 else "")
        print(f"  {x['confidence']:7.2f} {x['n']:5d} {x['accuracy']:9.3f} "
              f"{x['gap']:+8.3f}{flag}")

    e, m = ece(rows), mce(rows)
    print(f"\n  ECE {e:.4f}   (target < 0.05)   MCE {m:.4f}")

    mapping = fit(rows)
    print("\n== Recalibrated: stated confidence replaced by observed accuracy ==")
    for raw in sorted(mapping, reverse=True):
        print(f"  {raw:.2f} -> {mapping[raw]:.3f}")

    # ECE after applying the fit, on the same data the fit came from: an
    # in-sample floor, not a generalisation estimate.
    recal = [{"fields": {k: {**d, "confidence": apply_fit(mapping, d["confidence"])}
                         for k, d in r["fields"].items()}} for r in rows]
    print(f"\n  ECE after recalibration (in-sample): {ece(recal):.4f}")
    print("  Held-out ECE needs a second labelled run; this is a floor.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/confidence_results_v3.json")
