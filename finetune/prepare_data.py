"""Build the Donut fine-tuning manifest from USD-filtered WildReceipt, reusing
the gold reconstruction in distill.data. Writes finetune/data/{train,test}.jsonl
with {image, target} per line, plus special_tokens.json for the tokenizer.

Price fields are normalized (currency symbols / OCR noise stripped) so the
fine-tuned model outputs clean numbers. Text fields (store, date) keep the raw
gold because its concatenation noise cannot be safely undone.

Run:  python -m finetune.prepare_data
"""
import json
import os
import re

from distill.data import _records, _build_gold, _build_line_items, _is_usd
from finetune.donut_format import to_sequence, from_sequence, special_tokens

OUT = "finetune/data"
PRICE_KEYS = ("subtotal_price", "tax_price", "tip_price", "total_price")


def _clean_price(v):
    if not v:
        return v
    t = re.sub(r"[^\d.,]", "", str(v))
    return t or None


def build(split_file, usd_only=True, items=True):
    """items=False drops line items from the targets.

    Line-item tags outnumber header tags about 5:1 (4,476 <s_description> vs
    731 <s_store_name>), so with them included the model learns the frequent
    tokens and never emits the header fields. Header-only targets are also what
    the zero-shot baseline is scored on, which keeps the comparison honest.
    """
    rows = []
    for rec in _records(split_file):
        gold = _build_gold(rec)
        if usd_only and not _is_usd(rec, gold):
            continue
        for k in PRICE_KEYS:
            gold[k] = _clean_price(gold.get(k))
        line_items = []
        if items:
            line_items = _build_line_items(rec)
            for it in line_items:
                it["price"] = _clean_price(it.get("price"))
        rows.append({"image": rec["file_name"],
                     "target": to_sequence(gold, line_items)})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-items", dest="items", action="store_false",
                    help="header fields only (matches the zero-shot eval scope)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    counts = {}
    for split, fn in [("train", "train.txt"), ("test", "test.txt")]:
        rows = build(fn, items=args.items)
        counts[split] = len(rows)
        with open(f"{OUT}/{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    toks = special_tokens()
    with open(f"{OUT}/special_tokens.json", "w", encoding="utf-8") as f:
        json.dump(toks, f, indent=2)

    print(f"train={counts['train']}  test={counts['test']}  "
          f"special_tokens={len(toks)}  line_items={args.items}")
    sample = build("test.txt", items=args.items)[0]
    print(f"\nsample target:\n{sample['target']}")
    print(f"\nparsed:\n{json.dumps(from_sequence(sample['target']), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
