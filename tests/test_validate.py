"""Tests for arithmetic validation, built from real failure cases in the
2026-07-16 confidence run."""
import pytest

from distill.score import _to_float
from distill.validate import apply_validation, check_arithmetic, CAP


def fields(sub, tax, tot, conf=1.0):
    return {
        "subtotal_price": {"value": sub, "confidence": conf},
        "tax_price": {"value": tax, "confidence": conf},
        "total_price": {"value": tot, "confidence": conf},
    }


# ---------- comma-decimal parsing (Vienna receipt regression) ----------

@pytest.mark.parametrize("raw,expected", [
    ("89,60", 89.60),
    ("98,50", 98.50),
    ("20,90", 20.90),
    ("1.234,56", 1234.56),   # dot thousands + comma decimal
    ("1,234.56", 1234.56),   # US thousands unchanged
    ("1,234", 1234.0),       # US thousands without cents
])
def test_comma_decimal(raw, expected):
    assert _to_float(raw) == pytest.approx(expected)


# ---------- arithmetic checks ----------

def test_consistent_passes():
    # Walmart: 22.04 + 0.47 = 22.51
    assert check_arithmetic(fields("22.04", "0.47", "22.51")) == set()


def test_mismatch_flags_all_three():
    # HOME HOUSE: 100.75 + 78.00 != 113.35 (tax grabbed from wrong line)
    assert check_arithmetic(fields("100.75", "78.00", "113.35")) == {
        "subtotal_price", "tax_price", "total_price"}


def test_aqua_wrong_tax_flagged():
    # Aqua: tax read as 40.50 (the total) instead of 4.50
    assert check_arithmetic(fields("36.00", "40.50", "40.50")) != set()


def test_invented_subtotal_flagged():
    # Costco: no subtotal line printed; model copied total
    assert check_arithmetic(fields("70.34", None, "70.34")) == {"subtotal_price"}


def test_subtotal_differs_from_total_without_tax_ok():
    # discount/tip receipts legitimately have subtotal != total with no tax line
    assert check_arithmetic(fields("50.00", None, "45.00")) == set()


def test_missing_fields_ok():
    assert check_arithmetic(fields(None, None, "10.00")) == set()
    assert check_arithmetic(fields(None, None, None)) == set()


def test_rounding_tolerance():
    assert check_arithmetic(fields("10.00", "0.83", "10.83")) == set()
    assert check_arithmetic(fields("10.00", "0.83", "10.82")) == set()  # 1c rounding


# ---------- confidence capping ----------

def test_apply_validation_caps_confidence():
    f = apply_validation(fields("100.75", "78.00", "113.35", conf=1.0))
    for k in ("subtotal_price", "tax_price", "total_price"):
        assert f[k]["confidence"] == CAP
        assert f[k]["validation_flag"] == "arithmetic"


def test_apply_validation_leaves_consistent_untouched():
    f = apply_validation(fields("22.04", "0.47", "22.51", conf=1.0))
    for k in ("subtotal_price", "tax_price", "total_price"):
        assert f[k]["confidence"] == 1.0
        assert "validation_flag" not in f[k]
