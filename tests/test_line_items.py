"""Tests for line-item scoring, item-sum voting, and the tip-aware /
item-exculpation validation logic. Fixtures come from manually image-verified
receipts (Cho Eun, Costco Omaha, Walmart)."""
import pytest

from distill.score import score_line_items, items_sum
from distill.validate import check_arithmetic

CHO_EUN_GOLD = [
    {"description": "BIBIM.OCTOPUT1", "quantity": None, "price": "$13.99"},
    {"description": "S-FOODP.CAKT1", "quantity": None, "price": "$14.99"},
    {"description": "PORKDUMPLINT1", "quantity": None, "price": "$8.99"},
    {"description": "LABEEFRIBT1", "quantity": None, "price": "$17.99"},
]


def fields(sub, tax, tot, tip=None, conf=1.0):
    f = {
        "subtotal_price": {"value": sub, "confidence": conf},
        "tax_price": {"value": tax, "confidence": conf},
        "total_price": {"value": tot, "confidence": conf},
        "tip_price": {"value": tip, "confidence": conf},
    }
    return f


# ---------- item matching ----------

def test_perfect_match_spacing_and_symbols():
    pred = [
        {"description": "BIBIM. OCTOPU T1", "quantity": None, "price": "13.99"},
        {"description": "S-FOOD P.CAK T1", "quantity": None, "price": "14.99"},
        {"description": "PORK DUMPLIN T1", "quantity": None, "price": "8.99"},
        {"description": "LA BEEF RIB T1", "quantity": None, "price": "17.99"},
    ]
    r = score_line_items(pred, CHO_EUN_GOLD)
    assert r["f1"] == 1.0


def test_wrong_price_not_matched():
    pred = [{"description": "BIBIM. OCTOPU T1", "quantity": None, "price": "13.89"}]
    r = score_line_items(pred, CHO_EUN_GOLD)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 4


def test_unrelated_name_same_price_not_matched():
    pred = [{"description": "COFFEE", "quantity": None, "price": "13.99"}]
    r = score_line_items(pred, CHO_EUN_GOLD)
    assert r["tp"] == 0


def test_missed_items_counted_as_fn():
    pred = [{"description": "BIBIM OCTOPU", "quantity": None, "price": "13.99"}]
    r = score_line_items(pred, CHO_EUN_GOLD)
    assert r["tp"] == 1 and r["fn"] == 3


def test_duplicate_gold_items_need_duplicate_preds():
    gold = [
        {"description": "DEPOSITFEE", "price": "0.05"},
        {"description": "DEPOSITFEE", "price": "0.05"},
    ]
    pred = [{"description": "DEPOSIT FEE", "price": "0.05"}]
    r = score_line_items(pred, gold)
    assert r["tp"] == 1 and r["fn"] == 1


# ---------- items_sum ----------

def test_items_sum():
    assert items_sum(CHO_EUN_GOLD) == pytest.approx(55.96)
    assert items_sum([]) is None
    assert items_sum([{"price": None}, {"price": "abc"}]) is None
    assert items_sum([{"price": "￥17.99"}]) == pytest.approx(17.99)  # OCR yen noise


# ---------- tip-aware arithmetic ----------

def test_tip_receipt_consistent():
    assert check_arithmetic(fields("50.00", "4.00", "60.00", tip="6.00")) == set()


def test_tip_receipt_mismatch_flags_present_fields():
    flags = check_arithmetic(fields("50.00", "4.00", "60.00", tip="2.00"))
    assert flags == {"subtotal_price", "tax_price", "tip_price", "total_price"}


def test_tip_only_no_tax():
    assert check_arithmetic(fields("40.00", None, "48.00", tip="8.00")) == set()


# ---------- item-sum exculpation ----------

def test_items_exculpate_subtotal():
    # Cho Eun scenario: model misreads total (60.46 vs printed 60.44), so
    # arithmetic fails; but items sum to exactly the reported subtotal,
    # so the subtotal is confirmed and blame shifts to the other fields.
    f = fields("55.96", "4.48", "60.46")
    flags = check_arithmetic(f, line_items=CHO_EUN_GOLD)
    assert "subtotal_price" not in flags
    assert "total_price" in flags


def test_items_never_add_flags():
    # Walmart scenario: arithmetic consistent, item sum differs legitimately
    # (discounts, deposits). Items must not introduce flags.
    f = fields("22.04", "0.47", "22.51")
    items = [{"description": "STUFF", "price": "24.59"}]
    assert check_arithmetic(f, line_items=items) == set()


def test_items_mismatch_keeps_subtotal_flagged():
    f = fields("55.96", "4.48", "60.46")
    items = [{"description": "X", "price": "10.00"}]
    assert "subtotal_price" in check_arithmetic(f, line_items=items)
