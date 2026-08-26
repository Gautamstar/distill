"""Tests for the configurable arithmetic blame policy.

When `subtotal + tax = total` fails one of them is wrong, but measurement shows
the subtotal is the usual culprit and tax/total are usually collateral. Which
fields get capped is therefore a policy dial trading automation against
delivered accuracy, and these pin each setting's behaviour.
"""
import pytest

from distill.validate import BLAME, check_arithmetic, apply_validation, CAP


def fields(sub, tax, tot, tip=None, conf=1.0):
    return {
        "subtotal_price": {"value": sub, "confidence": conf},
        "tax_price": {"value": tax, "confidence": conf},
        "total_price": {"value": tot, "confidence": conf},
        "tip_price": {"value": tip, "confidence": conf},
    }


# 50 + 4 != 60, so the check fires under every policy
BROKEN = ("50.00", "4.00", "60.00")


def test_default_policy_blames_every_present_price_field():
    assert check_arithmetic(fields(*BROKEN), policy="all") == {
        "subtotal_price", "tax_price", "total_price"}


def test_subtotal_only_policy_spares_the_collateral():
    assert check_arithmetic(fields(*BROKEN), policy="subtotal") == {"subtotal_price"}


def test_subtotal_total_policy():
    assert check_arithmetic(fields(*BROKEN), policy="subtotal_total") == {
        "subtotal_price", "total_price"}


def test_none_policy_disables_the_gate():
    assert check_arithmetic(fields(*BROKEN), policy="none") == set()


def test_unknown_policy_falls_back_to_the_safe_default():
    """A typo in configuration must not silently disable the gate."""
    assert check_arithmetic(fields(*BROKEN), policy="nonsense") == {
        "subtotal_price", "tax_price", "total_price"}


def test_absent_fields_are_never_blamed():
    # no tip on this receipt, so it cannot be the culprit
    assert "tip_price" not in check_arithmetic(fields(*BROKEN), policy="all")


def test_policy_does_not_affect_the_invented_subtotal_check():
    """Subtotal == total with no tax is a separate rule and must still fire.

    This is the Costco case the README screenshots; a loosened blame policy
    must not quietly stop catching it.
    """
    for name in BLAME:
        assert check_arithmetic(fields("70.34", None, "70.34"), policy=name) == {
            "subtotal_price"}, name


def test_consistent_receipt_is_untouched_under_every_policy():
    for name in BLAME:
        assert check_arithmetic(fields("22.04", "0.47", "22.51"), policy=name) == set(), name


def test_apply_validation_threads_the_policy_through():
    f = apply_validation(fields(*BROKEN), policy="subtotal")
    assert f["subtotal_price"]["confidence"] == CAP
    assert f["total_price"]["confidence"] == 1.0      # spared
    assert "validation_flag" not in f["total_price"]


def test_item_sums_still_exculpate_a_confirmed_subtotal():
    items = [{"description": "x", "price": "50.00"}]
    flags = check_arithmetic(fields(*BROKEN), line_items=items, policy="all")
    assert "subtotal_price" not in flags
    assert "total_price" in flags
