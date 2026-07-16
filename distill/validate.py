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
import re

from .score import _to_float

PRICE_FIELDS = ("subtotal_price", "tax_price", "total_price")
CAP = 0.5   # flagged fields drop below any sane auto-accept threshold
TOL = 0.015

DATE_SHAPES = [
    re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"),          # 12/30/2016, 10.07.13
    re.compile(r"\d{1,2}[\s/\-.]*[A-Za-z]{3,9}[\s/\-.,']*\d{2,4}"),  # 04NOV'17, 25 Dec 13
    re.compile(r"[A-Za-z]{3,9}[\s/\-.,]*\d{1,2}[\s,'\-]*\d{2,4}"),   # Dec 25, 2016
]


def plausible_date(value: str) -> bool:
    return any(p.search(str(value)) for p in DATE_SHAPES)


def check_arithmetic(fields: dict) -> dict:
    """Given {field: {value, confidence}}, return set of price fields to flag."""
    sub = _to_float(fields["subtotal_price"]["value"]) if fields["subtotal_price"]["value"] else None
    tax = _to_float(fields["tax_price"]["value"]) if fields["tax_price"]["value"] else None
    tot = _to_float(fields["total_price"]["value"]) if fields["total_price"]["value"] else None

    flags = set()
    # All three present: they must add up.
    if sub is not None and tax is not None and tot is not None:
        if abs(sub + tax - tot) > TOL:
            flags.update(PRICE_FIELDS)
    # Subtotal equal to total with no tax: the known invented-subtotal pattern.
    elif sub is not None and tot is not None and tax is None:
        if abs(sub - tot) <= TOL:
            flags.add("subtotal_price")
    return flags


def apply_validation(fields: dict) -> dict:
    """Cap confidence of fields implicated by a failed check. Mutates and returns."""
    for f in check_arithmetic(fields):
        if fields[f]["confidence"] > CAP:
            fields[f]["confidence"] = CAP
            fields[f]["validation_flag"] = "arithmetic"
    date = fields.get("date", {})
    if date.get("value") and not plausible_date(date["value"]):
        if date["confidence"] > CAP:
            date["confidence"] = CAP
            date["validation_flag"] = "date_shape"
    return fields
