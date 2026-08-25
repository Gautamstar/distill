"""Convert Distill gold records to Donut target sequences (json2token) and back.

Donut trains a Swin + MBart model to generate a flat token sequence directly
from the image (no OCR). Each field key becomes a pair of special tokens; the
line-item list is separated with <sep/>. The same field order is used at train
and inference so the model learns a canonical layout.

  {"store_name": "COSTCO", "total_price": "70.34",
   "line_items": [{"description": "BANANAS", "price": "1.47"}]}
->
  <s_receipt><s_store_name>COSTCO</s_store_name><s_total_price>70.34</s_total_price>
  <s_line_items><s_description>BANANAS</s_description><s_price>1.47</s_price></s_line_items>
  </s_receipt>
"""
import re

TASK_START = "<s_receipt>"
TASK_END = "</s_receipt>"
SEP = "<sep/>"

FIELD_ORDER = ["store_name", "date", "subtotal_price", "tax_price",
               "tip_price", "total_price"]
ITEM_KEYS = ["description", "quantity", "price"]


def _tok(key):
    return f"<s_{key}>", f"</s_{key}>"


def special_tokens():
    toks = {TASK_START, TASK_END, SEP}
    for k in FIELD_ORDER + ["line_items"] + ITEM_KEYS:
        a, b = _tok(k)
        toks.update({a, b})
    return sorted(toks)


def to_sequence(gold: dict, line_items=None) -> str:
    parts = [TASK_START]
    for k in FIELD_ORDER:
        v = gold.get(k)
        if v:
            a, b = _tok(k)
            parts.append(f"{a}{v}{b}")
    items = line_items or []
    if items:
        a, b = _tok("line_items")
        segs = []
        for it in items:
            seg = []
            for ik in ITEM_KEYS:
                iv = it.get(ik)
                if iv:
                    ia, ib = _tok(ik)
                    seg.append(f"{ia}{iv}{ib}")
            if seg:
                segs.append("".join(seg))
        if segs:
            parts.append(a + SEP.join(segs) + b)
    parts.append(TASK_END)
    return "".join(parts)


def _extract(seq, key, dotall=False):
    """Value between an opening tag and whatever ends it.

    Deliberately does not require the matching closing tag: a fine-tuned model
    routinely emits `<s_subtotal_price> 55.96</s_total_price>`, having correctly
    identified the field and read the value while picking the wrong closer.
    Requiring the exact pair discards a correct extraction over a formatting
    artifact, so terminate on the next tag of any kind instead.
    """
    a, b = _tok(key)
    exact = re.search(re.escape(a) + r"(.*?)" + re.escape(b), seq,
                      re.DOTALL if dotall else 0)
    if exact:
        return exact.group(1).strip()
    loose = re.search(re.escape(a) + r"\s*([^<]*)", seq)
    return loose.group(1).strip() or None if loose else None


def from_sequence(seq: str) -> dict:
    """Parse a generated Donut sequence back into a Distill record."""
    out = {k: (_extract(seq, k) or None) for k in FIELD_ORDER}
    items_block = _extract(seq, "line_items", dotall=True)
    items = []
    if items_block:
        for seg in items_block.split(SEP):
            it = {ik: (_extract(seg, ik) or None) for ik in ITEM_KEYS}
            if any(it.values()):
                items.append(it)
    out["line_items"] = items
    return out
