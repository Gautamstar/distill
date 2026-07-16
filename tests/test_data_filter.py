"""Tests for the non-USD receipt filter, from real leaks found in confidence
runs: currency codes glued to digits ("AED105.00") and dot-normalized gold on
European receipts whose annotations still contain a standalone currency token."""
from distill.data import _is_usd


def rec(*texts):
    return {"annotations": [{"text": t, "label": 25} for t in texts]}


GOLD_EMPTY = {"store_name": None, "date": None, "subtotal_price": None,
              "tax_price": None, "total_price": None}


def test_us_receipt_passes():
    assert _is_usd(rec("WAL*MART", "SUBTOTAL", "22.04", "TAX 1", "TOTAL", "22.51"),
                   GOLD_EMPTY)


def test_currency_code_glued_to_digits():
    assert not _is_usd(rec("Tent Jumeirah", "AED105.00"), GOLD_EMPTY)   # Dubai leak
    assert not _is_usd(rec("ILIO'S", "EUR129,75"), GOLD_EMPTY)


def test_standalone_currency_token():
    assert not _is_usd(rec("Rechnungs-Summe in EUR", "89,60"), GOLD_EMPTY)  # Vienna leak
    assert not _is_usd(rec("GBP", "5.33"), GOLD_EMPTY)


def test_symbols():
    assert not _is_usd(rec("Total", "£23.95"), GOLD_EMPTY)
    assert not _is_usd(rec("Summe", "€12.00"), GOLD_EMPTY)


def test_currency_inside_words_not_matched():
    # "aed" in "read", "rp" in "sharp" etc. must not trigger the filter
    assert _is_usd(rec("PLEASE READ YOUR RECEIPT", "SHARP CHEDDAR", "EUROPEAN WEAR"),
                   GOLD_EMPTY)


def test_comma_decimal_gold_price_rejected():
    gold = dict(GOLD_EMPTY, total_price="20,90")
    assert not _is_usd(rec("GASTROBAR"), gold)
