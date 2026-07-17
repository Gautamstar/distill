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
from distill.score import aggregate, _match, items_sum, score_line_items
from distill.validate import apply_validation


def vote_line_items(preds):
    """Self-consistency for line items: majority-vote the SUM of item prices
    (the scalar validation consumes), then take the item list from a sample in
    the majority cluster. Returns (items, sum, confidence)."""
    sums = [items_sum(p.get("line_items")) for p in preds]
    clusters = []   # [sum_value, count, first_sample_index]
    for idx, v in enumerate(sums):
        for c in clusters:
            same = (v is None and c[0] is None) or \
                   (v is not None and c[0] is not None and abs(v - c[0]) < 0.01)
            if same:
                c[1] += 1
                break
        else:
            clusters.append([v, 1, idx])
    clusters.sort(key=lambda c: c[1], reverse=True)
    best_sum, count, rep = clusters[0]
    return preds[rep].get("line_items") or [], best_sum, count / len(preds)


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
    item_agg = {"tp": 0, "fp": 0, "fn": 0}

    t0 = time.time()
    for i, s in enumerate(samples):
        preds = extract_samples(s.image, k=args.k)
        items, isum, iconf = vote_line_items(preds)
        agg = apply_validation(aggregate(preds, TARGET_FIELDS),
                               line_items=items if iconf >= 0.6 else None)
        iscore = score_line_items(items, s.gold_items or [])
        for k2 in ("tp", "fp", "fn"):
            item_agg[k2] += iscore[k2]
        row = {"file": s.file_name, "fields": {},
               "line_items": {"pred": items, "gold": s.gold_items,
                              "sum": isum, "sum_conf": round(iconf, 2),
                              "f1": iscore["f1"]}}
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
              "  ".join(f"{f.split('_')[0]}={agg[f]['confidence']:.1f}" for f in TARGET_FIELDS) +
              f"  items_f1={iscore['f1']:.2f}", flush=True)
    elapsed = time.time() - t0

    ip = item_agg["tp"] / (item_agg["tp"] + item_agg["fp"]) if (item_agg["tp"] + item_agg["fp"]) else 0
    ir = item_agg["tp"] / (item_agg["tp"] + item_agg["fn"]) if (item_agg["tp"] + item_agg["fn"]) else 0
    if1 = 2 * ip * ir / (ip + ir) if (ip + ir) else 0

    print("\n========= CONFIDENCE vs CORRECTNESS =========")
    for b in ("accept", "flag"):
        t, c = buckets[b]["total"], buckets[b]["correct"]
        acc = c / t if t else 0.0
        label = "conf == 1.0" if b == "accept" else "conf <  1.0"
        print(f"{b:7s} ({label}):  n={t:4d}  accuracy={acc:.3f}")
    total = buckets["accept"]["total"] + buckets["flag"]["total"]
    auto = buckets["accept"]["total"] / total if total else 0.0
    print(f"auto-accept rate = {auto:.3f}  ({buckets['accept']['total']}/{total} fields)")
    print(f"line items: micro F1={if1:.3f}  P={ip:.3f}  R={ir:.3f}  "
          f"(tp={item_agg['tp']} fp={item_agg['fp']} fn={item_agg['fn']})")
    print(f"time={elapsed:.1f}s  ({elapsed / max(len(samples), 1):.1f}s/doc at k={args.k})")
    print("=============================================")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "buckets": buckets}, fh, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
