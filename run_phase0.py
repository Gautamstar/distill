"""Baseline: extract fields from a batch of WildReceipt receipts and score
field-level F1 against the gold labels. Prints the baseline number.

Run:  python run_phase0.py --n 100
"""
import argparse
import json
import os
import time

from distill.data import load_wildreceipt_samples
from distill.extract import extract
from distill.schema import TARGET_FIELDS
from distill.score import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of receipts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/phase0_results.json")
    args = ap.parse_args()

    print(f"Loading {args.n} WildReceipt receipts (split={args.split})...")
    samples = load_wildreceipt_samples(split=args.split, n=args.n)

    preds, golds, rows = [], [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        pred = extract(s.image).model_dump()
        preds.append(pred)
        golds.append(s.gold)
        rows.append({"file": s.file_name, "pred": pred, "gold": s.gold})
        print(f"[{i + 1}/{len(samples)}] pred={pred} gold={s.gold}")
    elapsed = time.time() - t0

    results = score(preds, golds, TARGET_FIELDS)

    print("\n================ BASELINE ================")
    for f in TARGET_FIELDS:
        m = results["per_field"][f]
        print(f"{f:16s} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    o = results["overall"]
    print(f"{'OVERALL':16s} F1={o['f1']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}")
    print(f"docs={len(samples)}  time={elapsed:.1f}s  "
          f"({elapsed / max(len(samples), 1):.1f}s/doc)")
    print("==========================================")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "rows": rows}, fh, indent=2, ensure_ascii=False)
    print(f"Saved detailed results to {args.out}")


if __name__ == "__main__":
    main()
