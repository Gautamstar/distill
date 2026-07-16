"""Scorer unit tests. The scorer is the foundation of every metric we publish;
these tests pin the behaviors that past error analysis established:
  - price matching tolerates $/commas and bare-decimal forms (".70")
  - text matching tolerates WildReceipt gold quirks (concatenated tokens,
    duplicated values, partial store names)
  - the "null"-string bug stays fixed at the score level (None vs value)
  - aggregate() clusters equivalent values when computing confidence
"""
import pytest

from distill.score import _to_float, _match, score, aggregate


# ---------- price normalization ----------

@pytest.mark.parametrize("raw,expected", [
    ("$26.09", 26.09),
    ("26.09", 26.09),
    (".70", 0.70),
    ("$0.00", 0.00),
    ("1,234.56", 1234.56),
    ("12", 12.0),
])
def test_to_float_parses(raw, expected):
    assert _to_float(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "abc", None, "$", "."])
def test_to_float_rejects_junk(raw):
    assert _to_float(raw) is None


# ---------- price field matching ----------

@pytest.mark.parametrize("pred,gold", [
    ("6.69", "$6.69"),
    ("0.70", ".70"),
    ("58.92", "58.92"),
    ("26.09", "$26.09"),
    (None, None),
])
def test_price_match(pred, gold):
    assert _match("total_price", pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("60.46", "$60.44"),   # the two-cent misread must count as wrong
    (None, "5.58"),
    ("5.58", None),
    ("5.88", "5.58"),      # computed instead of read
])
def test_price_mismatch(pred, gold):
    assert not _match("total_price", pred, gold)


# ---------- text field matching (store_name / date) ----------

@pytest.mark.parametrize("pred,gold", [
    # gold concatenates words
    ("CHO EUN KOREAN RESTAURANT", "CHOEUN KOREANRESTAURANT"),
    # gold has only part of the name (containment, either direction)
    ("COSTCO WHOLESALE", "COSTCO"),
    ("Aqua", "Aqwa aqua"),
    # gold duplicates the value
    ("07/05/2017", "07/05/2017 07/05/2017"),
    # gold appends junk
    ("12/30/2016", "12/30/2016FRI"),
    ("metro", "metro"),
])
def test_text_match(pred, gold):
    assert _match("store_name", pred, gold)


@pytest.mark.parametrize("pred,gold", [
    ("26/03/2015", "26/03/2016"),   # real one-digit date misread
    (None, "COSTCO"),
    ("COSTCO", None),
    ("ab", "abcdef"),               # containment needs >= 3 chars
])
def test_text_mismatch(pred, gold):
    assert not _match("date", pred, gold)


# ---------- end-to-end scoring ----------

def test_score_counts():
    preds = [
        {"total_price": "10.00", "tax_price": "1.00"},   # tp, tp
        {"total_price": "20.00", "tax_price": None},     # tp, fn (gold has tax)
        {"total_price": "99.99", "tax_price": "5.00"},   # fp+fn (wrong), fp (gold null)
        {"total_price": None, "tax_price": None},        # both null-null: skipped
    ]
    golds = [
        {"total_price": "$10.00", "tax_price": "1.00"},
        {"total_price": "20.00", "tax_price": "2.00"},
        {"total_price": "30.00", "tax_price": None},
        {"total_price": None, "tax_price": None},
    ]
    r = score(preds, golds, fields=["total_price", "tax_price"])
    assert r["per_field"]["total_price"]["tp"] == 2
    assert r["per_field"]["total_price"]["fp"] == 1
    assert r["per_field"]["total_price"]["fn"] == 1
    assert r["per_field"]["tax_price"]["tp"] == 1
    assert r["per_field"]["tax_price"]["fp"] == 1
    assert r["per_field"]["tax_price"]["fn"] == 1
    assert r["overall"]["tp"] == 3


def test_perfect_score_is_one():
    rows = [{"total_price": "5.00"}, {"total_price": None}]
    r = score(rows, rows, fields=["total_price"])
    assert r["overall"]["f1"] == 1.0


# ---------- self-consistency aggregation ----------

def test_aggregate_majority_and_confidence():
    samples = [
        {"total_price": "10.00"},
        {"total_price": "$10.00"},   # equivalent form joins the same cluster
        {"total_price": "10.00"},
        {"total_price": "12.00"},
        {"total_price": "10.00"},
    ]
    agg = aggregate(samples, fields=["total_price"])
    assert _match("total_price", agg["total_price"]["value"], "10.00")
    assert agg["total_price"]["confidence"] == pytest.approx(0.8)


def test_aggregate_unanimous_none():
    samples = [{"tax_price": None}] * 5
    agg = aggregate(samples, fields=["tax_price"])
    assert agg["tax_price"]["value"] is None
    assert agg["tax_price"]["confidence"] == 1.0


def test_aggregate_disagreement_lowers_confidence():
    samples = [
        {"subtotal_price": "5.00"},
        {"subtotal_price": None},
        {"subtotal_price": "5.00"},
        {"subtotal_price": "6.00"},
        {"subtotal_price": None},
    ]
    agg = aggregate(samples, fields=["subtotal_price"])
    assert agg["subtotal_price"]["confidence"] == pytest.approx(0.4)
