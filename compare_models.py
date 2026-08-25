"""Compare the zero-shot VLM and the fine-tuned Donut on identical terms.

Both are re-scored here from saved predictions over the same field set with the
same scorer. Reporting one on five fields and the other on six - which is what
happened when the Donut evaluation picked up `tip_price` and the baseline did
not - makes the fine-tune look worse than it is for no reason.

Run:  python compare_models.py
"""
import json
from pathlib import Path

from distill.score import score

OUT = Path("outputs")
# The field set the published baseline uses. tip_price is excluded: it appears
# in 61 of 1,112 training receipts, too rare to score meaningfully, and the
# zero-shot number was never measured on it.
FIELDS = ["store_name", "date", "subtotal_price", "tax_price", "total_price"]


def _load(path, pred_key="pred", gold_key="gold"):
    d = json.load(open(path, encoding="utf-8"))
    return ([r[pred_key] for r in d["rows"]], [r[gold_key] for r in d["rows"]])


def main():
    vlm_p, vlm_g = _load(OUT / "phase0_results.json")
    don_p, don_g = _load(OUT / "donut_eval.json")

    vlm = score(vlm_p, vlm_g, FIELDS)
    don = score(don_p, don_g, FIELDS)

    print("Same 100-receipt test split, same scorer, same fields.\n")
    print(f"  {'field':16s} {'zero-shot':>10s} {'fine-tuned':>11s} {'delta':>8s}")
    for f in FIELDS:
        a, b = vlm["per_field"][f]["f1"], don["per_field"][f]["f1"]
        print(f"  {f:16s} {a:10.3f} {b:11.3f} {b - a:+8.3f}")
    a, b = vlm["overall"]["f1"], don["overall"]["f1"]
    print(f"  {'OVERALL':16s} {a:10.3f} {b:11.3f} {b - a:+8.3f}")
    print(f"\n  gap to parity: {a - b:.3f}")


if __name__ == "__main__":
    main()
