"""Tests for the Donut serving runtime.

Parsing is the part that decides whether a correct read survives to the API, so
it is tested against the exact malformed shapes the model actually produces.
"""
import pytest

from distill.donut_runtime import parse_sequence, _value


def test_well_formed_sequence():
    s = ("<s_receipt><s_store_name>COSTCO</s_store_name>"
         "<s_total_price>70.34</s_total_price></s_receipt>")
    r = parse_sequence(s)
    assert r["store_name"] == "COSTCO"
    assert r["total_price"] == "70.34"
    assert r["date"] is None


def test_mismatched_closing_tag_still_yields_the_value():
    # the model's most common formatting error: right field, wrong closer
    s = "<s_receipt><s_subtotal_price> 55.96</s_total_price></s_receipt>"
    assert parse_sequence(s)["subtotal_price"] == "55.96"


def test_leading_space_is_stripped():
    assert _value("<s_date> 12/30/2016</s_date>", "date") == "12/30/2016"


def test_absent_field_is_none_not_empty_string():
    r = parse_sequence("<s_receipt><s_store_name>X</s_store_name></s_receipt>")
    assert r["tax_price"] is None


def test_empty_tag_is_none():
    assert _value("<s_tax_price></s_tax_price>", "tax_price") is None


def test_truncated_generation_does_not_raise():
    r = parse_sequence("<s_receipt><s_store_name>COST")
    assert r["store_name"] == "COST"


def test_garbage_yields_all_none():
    r = parse_sequence("no tags at all")
    assert all(r[k] is None for k in r if k != "line_items")


def test_line_items_always_present_and_empty():
    """The fine-tune is header-only; it must not imply item support."""
    assert parse_sequence("<s_receipt></s_receipt>")["line_items"] == []


class _Tok:
    eos_token = "</s>"
    pad_token = "<pad>"


def test_special_tokens_are_stripped():
    s = "<s_receipt><s_store_name>ACME</s_store_name></s_receipt></s><pad><pad>"
    assert parse_sequence(s, _Tok())["store_name"] == "ACME"


def test_output_shape_matches_the_vlm_backend():
    from distill.schema import TARGET_FIELDS
    r = parse_sequence("<s_receipt></s_receipt>")
    assert set(r) == set(TARGET_FIELDS) | {"line_items"}
