"""Print reconstructed gold line items for receipts whose images we have
manually verified, so the y-alignment pairing can be checked before anything
is built on top of it.

Run:  python inspect_line_items.py [file_substring ...]
"""
import json
import sys

from distill.data import _build_line_items, _records, BASE

# Receipts manually verified against their images in the 2026-07-16 session.
DEFAULT_TARGETS = [
    "845be0dd",   # CHO EUN: 4 items 13.99 / 14.99 / 8.99 / 17.99, subtotal 55.96
    "4381",       # Costco Omaha: 11 produce items, total 70.34
    "96ae21a2",   # Walmart: ~20 items incl. deposits/fees, subtotal 22.04
]


def main(targets):
    for split_file in ("test.txt", "train.txt"):
        for rec in _records(split_file):
            if not any(t in rec["file_name"] for t in targets):
                continue
            items = _build_line_items(rec)
            total = sum_ok = 0.0
            print(f"\n===== {rec['file_name']} ({len(items)} items) =====")
            for it in items:
                q = f" x{it['quantity']}" if it["quantity"] else ""
                print(f"  {it['description'][:40]:42s}{q:8s} {it['price']}")
            try:
                s = sum(float(str(i["price"]).replace("$", "").replace(",", ""))
                        for i in items)
                print(f"  {'SUM OF ITEM PRICES':42s}{'':8s} {s:.2f}")
            except ValueError:
                print("  (sum unavailable: non-numeric price present)")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_TARGETS)
