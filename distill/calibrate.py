"""Calibration of the confidence signal.

Confidence is only useful if the number means what it says: fields marked 0.6
should be right about 60% of the time. Separation (confident fields being more
accurate than flagged ones) is necessary but not sufficient - a signal can rank
correctly and still be badly scaled, which makes any threshold policy guesswork.

Expected Calibration Error (ECE) is the average gap between stated confidence
and observed accuracy, weighted by how many fields sit at each level. Maximum
Calibration Error (MCE) is the worst single gap.

Because confidence comes from k-sample agreement (and a fixed validation cap),
values are discrete, so each distinct value is its own bin - no arbitrary
bucketing.
"""
from collections import defaultdict


def bins(rows, fields=None):
    """[{confidence, n, accuracy, gap}] over (confidence, correct) observations."""
    by_conf = defaultdict(list)
    for r in rows:
        for name, d in r["fields"].items():
            if fields and name not in fields:
                continue
            by_conf[round(float(d["confidence"]), 3)].append(bool(d["correct"]))
    out = []
    for conf in sorted(by_conf, reverse=True):
        oks = by_conf[conf]
        acc = sum(oks) / len(oks)
        out.append({"confidence": conf, "n": len(oks),
                    "accuracy": round(acc, 4), "gap": round(acc - conf, 4)})
    return out


def ece(rows, fields=None):
    b = bins(rows, fields)
    total = sum(x["n"] for x in b)
    if not total:
        return 0.0
    return round(sum(x["n"] / total * abs(x["gap"]) for x in b), 4)


def mce(rows, fields=None):
    b = bins(rows, fields)
    return round(max((abs(x["gap"]) for x in b), default=0.0), 4)


def fit(rows, fields=None, min_n=5):
    """Map each raw confidence to the accuracy actually observed at it.

    Levels with fewer than `min_n` observations keep their raw value: fitting a
    correction on a handful of samples would encode noise. Deliberately a plain
    lookup rather than a smooth fit - the inputs are discrete and few, and a
    lookup is inspectable, which matters more here than elegance.
    """
    return {x["confidence"]: x["accuracy"] for x in bins(rows, fields)
            if x["n"] >= min_n}


def apply_fit(mapping, confidence):
    """Calibrated value for a raw confidence, nearest known level if unseen."""
    c = round(float(confidence), 3)
    if c in mapping:
        return mapping[c]
    if not mapping:
        return c
    return mapping[min(mapping, key=lambda k: abs(k - c))]
