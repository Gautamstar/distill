"""Confidence scoring via self-consistency: sample the extraction k times per
receipt, set each field's value to the majority and its confidence to the
agreement fraction, then test whether low confidence predicts errors.

The key question: are fully-confident fields (all k samples agree) much more
accurate than flagged ones? If so, we can auto-accept the confident ones and
route the rest to human review.

Run:  python run_confidence.py --n 30 --k 5
"""
import argparse
import json
import os
import time

from distill.data import load_wildreceipt_samples
from distill.extract import extract_samples
from distill.schema import TARGET_FIELDS
from distill.score import aggregate, _match
from distill.validate import apply_validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=5, help="samples per receipt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/confidence_results.json")
    args = ap.parse_args()

    samples = load_wildreceipt_samples(split=args.split, n=args.n)
    rows = []
    buckets = {"accept": {"correct": 0, "total": 0}, "flag": {"correct": 0, "total": 0}}

    t0 = time.time()
    for i, s in enumerate(samples):
        preds = extract_samples(s.image, k=args.k)
        agg = apply_validation(aggregate(preds, TARGET_FIELDS))
        row = {"file": s.file_name, "fields": {}}
        for f in TARGET_FIELDS:
            val, conf = agg[f]["value"], agg[f]["confidence"]
            correct = _match(f, val, s.gold.get(f))
            row["fields"][f] = {"value": val, "confidence": round(conf, 2),
                                "gold": s.gold.get(f), "correct": correct}
            b = "accept" if conf >= 0.999 else "flag"      # all k agree = auto-accept
            buckets[b]["total"] += 1
            buckets[b]["correct"] += int(correct)
        rows.append(row)
        print(f"[{i + 1}/{len(samples)}] " +
              "  ".join(f"{f.split('_')[0]}={agg[f]['confidence']:.1f}" for f in TARGET_FIELDS))
    elapsed = time.time() - t0

    print("\n========= CONFIDENCE vs CORRECTNESS =========")
    for b in ("accept", "flag"):
        t, c = buckets[b]["total"], buckets[b]["correct"]
        acc = c / t if t else 0.0
        label = "conf == 1.0" if b == "accept" else "conf <  1.0"
        print(f"{b:7s} ({label}):  n={t:4d}  accuracy={acc:.3f}")
    total = buckets["accept"]["total"] + buckets["flag"]["total"]
    auto = buckets["accept"]["total"] / total if total else 0.0
    print(f"auto-accept rate = {auto:.3f}  ({buckets['accept']['total']}/{total} fields)")
    print(f"time={elapsed:.1f}s  ({elapsed / max(len(samples), 1):.1f}s/doc at k={args.k})")
    print("=============================================")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "buckets": buckets}, fh, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
