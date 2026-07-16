"""Re-score saved predictions after a scorer change, without re-running the model.

Run:  python rescore.py
"""
import json

from distill.schema import TARGET_FIELDS
from distill.score import score


def main(path: str = "outputs/phase0_results.json"):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    preds = [r["pred"] for r in rows]
    golds = [r["gold"] for r in rows]
    results = score(preds, golds, TARGET_FIELDS)

    print(f"Re-scored {len(rows)} saved predictions:\n")
    for f in TARGET_FIELDS:
        m = results["per_field"][f]
        print(f"{f:16s} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    o = results["overall"]
    print(f"{'OVERALL':16s} F1={o['f1']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}")


if __name__ == "__main__":
    main()
