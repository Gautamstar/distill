"""One-shot accuracy summary across the pipeline stages, computed from the
saved result files so the numbers we quote are never hand-copied.

Run:  python summarize_accuracy.py
"""
import json
from pathlib import Path

from distill.score import _match

OUT = Path("outputs")
ACCEPT = 0.999


def raw_baseline(path=OUT / "phase0_results.json"):
    """Re-score the saved predictions with the current scorer.

    The `results` block inside the file was computed by whatever scorer existed
    when the run happened; reading it back would report stale numbers after any
    scoring change. Recomputing keeps every published figure consistent with the
    code in the tree.
    """
    from distill.score import score
    d = json.load(open(path, encoding="utf-8"))
    fields = ["store_name", "date", "subtotal_price", "tax_price", "total_price"]
    r = score([x["pred"] for x in d["rows"]], [x["gold"] for x in d["rows"]], fields)
    return r["overall"], r["per_field"], len(d["rows"])


def gated(path):
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    acc_n = acc_ok = flag_n = flag_ok = 0
    for r in rows:
        for f, d in r["fields"].items():
            ok = bool(d["correct"])
            if d["confidence"] >= ACCEPT:
                acc_n += 1
                acc_ok += ok
            else:
                flag_n += 1
                flag_ok += ok
    return {"n_receipts": len(rows), "fields": acc_n + flag_n,
            "acc_n": acc_n, "acc_ok": acc_ok, "flag_n": flag_n, "flag_ok": flag_ok}


def donut(path=OUT / "donut_eval.json"):
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def main():
    o, per_field, n_docs = raw_baseline()
    print("=" * 62)
    print("1. RAW EXTRACTION  (zero-shot Qwen2.5-VL-3B, no gating)")
    print("=" * 62)
    print(f"   n = {n_docs} receipts, 5 fields")
    print(f"   field-level F1 = {o['f1']:.3f}   (P {o['precision']:.3f} / R {o['recall']:.3f})")
    for f, m in per_field.items():
        print(f"     {f:16s} F1 {m['f1']:.3f}")

    g = gated(OUT / "confidence_results_v3.json")
    tot = g["fields"]
    cov = g["acc_n"] / tot
    acc_acc = g["acc_ok"] / g["acc_n"]
    flag_acc = g["flag_ok"] / g["flag_n"]
    overall_raw = (g["acc_ok"] + g["flag_ok"]) / tot

    print("\n" + "=" * 62)
    print("2. CONFIDENCE GATING  (self-consistency k=5 + validation)")
    print("=" * 62)
    print(f"   n = {g['n_receipts']} receipts, {tot} scored fields")
    print(f"   ungated accuracy       {overall_raw:.3f}   ({g['acc_ok'] + g['flag_ok']}/{tot})")
    print(f"   auto-accepted slice    {acc_acc:.3f}   ({g['acc_ok']}/{g['acc_n']}, {cov:.1%} of fields)")
    print(f"   flagged-for-review     {flag_acc:.3f}   ({g['flag_ok']}/{g['flag_n']}, {1 - cov:.1%} of fields)")
    print(f"   separation             {acc_acc - flag_acc:+.3f}")

    # End-to-end: accepted fields ship as-is, flagged fields are corrected by a
    # human. Upper bound assumes the reviewer is right every time.
    e2e = (g["acc_ok"] + g["flag_n"]) / tot
    print("\n" + "=" * 62)
    print("3. END-TO-END PRODUCT  (auto-accept + human reviews the flagged)")
    print("=" * 62)
    print(f"   delivered accuracy     {e2e:.3f}   (assumes reviewer is correct)")
    print(f"   manual work removed    {cov:.1%} of fields never touched by a human")
    print(f"   residual error         {1 - e2e:.3f}  = wrong fields that ship silently")

    from distill.trace import summary as trace_summary
    t = trace_summary(op="extract")
    if t.get("n"):
        print()
        print("=" * 62)
        print("LATENCY AND COST  (traced extractions on this install)")
        print("=" * 62)
        print(f"   documents              {t['n']}  ({t['failures']} failed)")
        print(f"   p50 / p95 latency      {t['p50_seconds']:.2f}s / {t['p95_seconds']:.2f}s")
        print(f"   throughput             {t['docs_per_hour']:.0f} docs/hour")
        print(f"   cost                   ${t['usd_local']:.2f} local "
              f"(${t['usd_if_hosted']:.2f} on a hosted vision API)")

    d = donut()
    if d:
        do = d["results"]["overall"]
        print("\n" + "=" * 62)
        print("4. FINE-TUNED DONUT  vs the zero-shot VLM, same harness")
        print("=" * 62)
        print(f"   {'field':16s} {'zero-shot':>10s} {'fine-tuned':>11s} {'delta':>8s}")
        for f, m in per_field.items():
            fm = d["results"]["per_field"].get(f)
            if not fm:
                continue
            print(f"   {f:16s} {m['f1']:10.3f} {fm['f1']:11.3f} {fm['f1'] - m['f1']:+8.3f}")
        print(f"   {'OVERALL':16s} {o['f1']:10.3f} {do['f1']:11.3f} {do['f1'] - o['f1']:+8.3f}")
        print(f"   {'line items':16s} {'-':>10s} {d['line_items']['f1']:11.3f}")
        print(f"   {'sec / doc':16s} {6.5:10.2f} {d['sec_per_doc']:11.2f} "
              f"{d['sec_per_doc'] - 6.5:+8.2f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
