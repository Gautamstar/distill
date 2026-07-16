"""Load WildReceipt receipts (MMOCR format) and build gold labels for our schema.

Data layout (from the MMOCR tarball, extracted to data/wildreceipt/):
  class_list.txt   label id -> category name
  train.txt        one JSON record per line
  test.txt         one JSON record per line
  image_files/...  the receipt images

Each record: {"file_name": ..., "annotations": [{"box": [...], "text": ..., "label": id}, ...]}
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

BASE = Path("data/wildreceipt")

# WildReceipt value-category id -> Distill field (ids from class_list.txt).
LABEL_TO_FIELD = {
    1: "store_name",       # Store_name_value
    7: "date",             # Date_value
    17: "subtotal_price",  # Subtotal_value
    19: "tax_price",       # Tax_value
    23: "total_price",     # Total_value
}
# Multi-token fields are joined; single-value price fields take the first box.
JOIN_FIELDS = {"store_name", "date"}

# WildReceipt is internationally sourced; keep only dollar (North American)
# receipts. Gold prices are sometimes dot-normalized by annotators (a Vienna
# receipt printing "89,60" had gold "89.60"), so checking price fields alone
# leaks foreign receipts: scan EVERY annotation text on the receipt for
# currency markers, plus comma decimals in the gold prices.
NON_USD = re.compile(r"(€|£|¥|\b(eur|euro|gbp|aed|rp|inr|sgd|myr|php|idr)\b)", re.I)
COMMA_DECIMAL = re.compile(r"\d,\d{2}(\D|$)")


@dataclass
class Sample:
    image: object      # PIL.Image.Image
    gold: dict          # {field: value or None}
    file_name: str


def _records(split_file: str):
    with open(BASE / split_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_gold(rec: dict) -> dict:
    buckets: dict = {}
    for ann in rec.get("annotations", []):
        field = LABEL_TO_FIELD.get(ann.get("label"))
        if field:
            text = str(ann.get("text", "")).strip()
            if text:
                buckets.setdefault(field, []).append(text)
    gold = {}
    for field in LABEL_TO_FIELD.values():
        texts = buckets.get(field, [])
        if not texts:
            gold[field] = None
        elif field in JOIN_FIELDS:
            gold[field] = " ".join(texts)
        else:
            gold[field] = texts[0]
    return gold


def _is_usd(rec: dict, gold: dict) -> bool:
    for ann in rec.get("annotations", []):
        if NON_USD.search(str(ann.get("text", ""))):
            return False
    for f in ("subtotal_price", "tax_price", "total_price"):
        v = gold.get(f)
        if v and COMMA_DECIMAL.search(str(v)):
            return False
    return True


def load_wildreceipt_samples(split: str = "test", n: Optional[int] = None,
                             usd_only: bool = True):
    split_file = "test.txt" if split == "test" else "train.txt"
    samples = []
    for rec in _records(split_file):
        gold = _build_gold(rec)
        if usd_only and not _is_usd(rec, gold):
            continue
        img = Image.open(BASE / rec["file_name"]).convert("RGB")
        samples.append(Sample(image=img, gold=gold, file_name=rec["file_name"]))
        if n and len(samples) >= n:
            break
    return samples
