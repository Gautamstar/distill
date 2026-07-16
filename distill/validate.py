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
"""
from .score import _to_float

PRICE_FIELDS = ("subtotal_price", "tax_price", "total_price")
CAP = 0.5   # flagged fields drop below any sane auto-accept threshold
TOL = 0.015


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
    """Cap confidence of arithmetically implicated fields. Mutates and returns."""
    for f in check_arithmetic(fields):
        if fields[f]["confidence"] > CAP:
            fields[f]["confidence"] = CAP
            fields[f]["validation_flag"] = "arithmetic"
    return fields
