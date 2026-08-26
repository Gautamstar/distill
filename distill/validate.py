"""Arithmetic validation of extracted price fields.

Self-consistency confidence catches stochastic errors (samples disagree) but
not systematic ones (the model reads the same wrong line every time). The two
systematic families found in error analysis are:
  - invented subtotal: subtotal reported equal to total on receipts with no
    subtotal line (and no tax found)
  - wrong-line grabs: a tax or total taken from an unrelated line, which
    usually breaks subtotal + tax = total

Arithmetic is an orthogonal signal that catches both. Fields implicated by a
failed check get their confidence capped so they route to human review.
A second check covers the date field: confident date errors in the 2026-07-16
run were non-date text read as a date ("TIME 22:55", phone/member numbers), so
a value that matches no plausible date shape gets flagged too.
"""
import os
import re

from .score import _to_float, items_sum

PRICE_FIELDS = ("subtotal_price", "tax_price", "tip_price", "total_price")
CAP = 0.5   # flagged fields drop below any sane auto-accept threshold
TOL = 0.015

# When the arithmetic check fires, one of the price fields is wrong - but which?
# Measured over 32 capped fields, 24 of which were actually correct:
#
#   field      capped  wrongly capped
#   subtotal     12      7  (58%)
#   tax          10      8  (80%)
#   total        10      9  (90%)
#
# The culprit is overwhelmingly the subtotal; tax and total are read off
# labelled lines and are usually collateral. Narrowing the blame automates more
# work but ships more unreviewed errors, and the exchange rate is real:
#
#   blame                coverage   accepted acc   delivered
#   all four (default)      59.3%         94.4%       96.7%
#   subtotal + total        66.0%         92.9%       95.3%
#   subtotal only           72.7%         92.7%       94.7%
#   none                    80.7%         89.3%       91.3%
#
# There is no free lunch here: neither sampling disagreement nor the size of the
# arithmetic gap separates the genuinely-wrong caps from the collateral, so the
# choice is a policy dial, not a bug to fix. The default stays conservative -
# highest delivered accuracy - and `DISTILL_BLAME` moves it when a deployment
# would rather buy automation with a lower accuracy floor.
# Reproduce with `python simulate_policies.py`.
BLAME = {
    "all": set(PRICE_FIELDS),
    "subtotal_total": {"subtotal_price", "total_price"},
    "subtotal": {"subtotal_price"},
    "none": set(),
}
DEFAULT_BLAME = os.environ.get("DISTILL_BLAME", "all")

DATE_SHAPES = [
    re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"),          # 12/30/2016, 10.07.13
    re.compile(r"\d{1,2}[\s/\-.]*[A-Za-z]{3,9}[\s/\-.,']*\d{2,4}"),  # 04NOV'17, 25 Dec 13
    re.compile(r"[A-Za-z]{3,9}[\s/\-.,]*\d{1,2}[\s,'\-]*\d{2,4}"),   # Dec 25, 2016
]


def plausible_date(value: str) -> bool:
    return any(p.search(str(value)) for p in DATE_SHAPES)


def _val(fields, name):
    v = fields.get(name, {}).get("value")
    return _to_float(v) if v else None


def _blame_set(policy=None):
    return BLAME.get(policy or DEFAULT_BLAME, BLAME["all"])


def check_arithmetic(fields: dict, line_items=None, policy=None) -> set:
    """Given {field: {value, confidence}}, return the set of price fields to flag.

    line_items, when provided, add a second equation: if the item prices sum
    to the reported subtotal, the subtotal is confirmed and is exculpated from
    any arithmetic flag (blame shifts to the other fields). Item sums never
    add flags: real receipts legitimately break them (discounts, deposits).
    """
    sub, tax, tot = _val(fields, "subtotal_price"), _val(fields, "tax_price"), _val(fields, "total_price")
    tip = _val(fields, "tip_price")

    flags = set()
    if sub is not None and tot is not None:
        if tax is None and tip is None:
            # Subtotal equal to total with no tax/tip: invented-subtotal pattern.
            if abs(sub - tot) <= TOL:
                flags.add("subtotal_price")
        elif abs(sub + (tax or 0) + (tip or 0) - tot) > TOL:
            blame = _blame_set(policy)
            flags.update(f for f in PRICE_FIELDS
                         if f in blame and fields.get(f, {}).get("value") is not None)

    if flags and line_items:
        s = items_sum(line_items)
        if s is not None and sub is not None and abs(s - sub) <= TOL:
            flags.discard("subtotal_price")
    return flags


def apply_validation(fields: dict, line_items=None, policy=None) -> dict:
    """Cap confidence of fields implicated by a failed check. Mutates and returns."""
    for f in check_arithmetic(fields, line_items=line_items, policy=policy):
        if fields[f]["confidence"] > CAP:
            fields[f]["confidence"] = CAP
            fields[f]["validation_flag"] = "arithmetic"
    date = fields.get("date", {})
    if date.get("value") and not plausible_date(date["value"]):
        if date["confidence"] > CAP:
            date["confidence"] = CAP
            date["validation_flag"] = "date_shape"
    return fields
