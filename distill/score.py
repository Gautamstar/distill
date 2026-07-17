"""Field-level scoring: precision, recall, F1 against gold labels.

Price fields are compared numerically (dollar amounts, tolerant of $ and commas);
text fields (store_name, date) are compared as normalized strings.
"""
import re
from difflib import SequenceMatcher

from .schema import TARGET_FIELDS

PRICE_FIELDS = {"subtotal_price", "tax_price", "tip_price", "total_price"}


def _to_float(s):
    t = re.sub(r"[^\d.,]", "", str(s))
    if not t:
        return None
    # European decimal comma ("89,60"): last comma followed by 1-2 trailing
    # digits acts as the decimal point; earlier dots/commas are separators.
    m = re.search(r",(\d{1,2})$", t)
    if m:
        head = t[: m.start()].replace(".", "").replace(",", "")
        t = head + "." + m.group(1)
    else:
        t = t.replace(",", "")
    if t.count(".") > 1:                      # keep only the last dot as the decimal
        parts = t.split(".")
        t = "".join(parts[:-1]) + "." + parts[-1]
    if t in ("", "."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _norm_str(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _text_match(pred, gold) -> bool:
    p, g = _norm_str(pred), _norm_str(gold)
    if not p or not g:
        return p == g
    if p == g:
        return True
    # WildReceipt gold concatenates/duplicates tokens or appends junk (store
    # "COSTCO" vs read "COSTCO WHOLESALE"; date "07/05/2017 07/05/2017"). Count
    # containment of the shorter string (>= 3 chars) in the longer as a match.
    short, long = (p, g) if len(p) <= len(g) else (g, p)
    return len(short) >= 3 and short in long


def _match(field: str, pred, gold) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    if field in PRICE_FIELDS:
        pf, gf = _to_float(pred), _to_float(gold)
        return pf is not None and gf is not None and abs(pf - gf) < 0.01
    return _text_match(pred, gold)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _get(row, field):
    return row.get(field) if isinstance(row, dict) else getattr(row, field, None)


def score(preds, golds, fields=TARGET_FIELDS) -> dict:
    per_field = {}
    agg = {"tp": 0, "fp": 0, "fn": 0}
    for f in fields:
        tp = fp = fn = 0
        for p, g in zip(preds, golds):
            pv, gv = _get(p, f), _get(g, f)
            if pv is None and gv is None:
                continue
            if _match(f, pv, gv):
                tp += 1
            else:
                if pv is not None:
                    fp += 1
                if gv is not None:
                    fn += 1
        per_field[f] = _prf(tp, fp, fn)
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn
    return {"per_field": per_field, "overall": _prf(agg["tp"], agg["fp"], agg["fn"])}


def _name_sim(a, b) -> float:
    return SequenceMatcher(None, _norm_str(a), _norm_str(b)).ratio()


def score_line_items(pred_items, gold_items) -> dict:
    """Item-level P/R/F1. A predicted item matches a gold item when the prices
    agree numerically and the descriptions are similar (gold OCR concatenates
    words, so similarity runs on space-stripped normalized text)."""
    used = set()
    tp = 0
    for g in gold_items:
        for j, p in enumerate(pred_items):
            if j in used:
                continue
            gp, pp = _to_float(g.get("price")), _to_float(p.get("price"))
            if gp is None or pp is None or abs(gp - pp) >= 0.01:
                continue
            if _name_sim(g.get("description") or "", p.get("description") or "") >= 0.4:
                used.add(j)
                tp += 1
                break
    return _prf(tp, len(pred_items) - tp, len(gold_items) - tp)


def items_sum(items):
    """Sum of parseable line prices, or None if no line has a parseable price."""
    vals = [_to_float(i.get("price")) for i in items or []]
    vals = [v for v in vals if v is not None]
    return round(sum(vals), 2) if vals else None


def aggregate(samples, fields=TARGET_FIELDS) -> dict:
    """Self-consistency aggregation over k sampled extractions.

    For each field, cluster the k values by agreement (reusing the field matcher),
    take the largest cluster as the value, and set confidence to that cluster's
    share of the k samples.
    """
    result = {}
    for f in fields:
        values = [_get(s, f) for s in samples]
        clusters = []   # [representative_value, count]
        for v in values:
            for c in clusters:
                if _match(f, v, c[0]):
                    c[1] += 1
                    break
            else:
                clusters.append([v, 1])
        clusters.sort(key=lambda c: c[1], reverse=True)
        best_val, best_count = clusters[0]
        result[f] = {"value": best_val, "confidence": best_count / len(values)}
    return result
