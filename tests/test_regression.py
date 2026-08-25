"""Regression gate on the metrics the README publishes.

CI has no GPU, so it cannot re-run the model. What it *can* do is re-score
frozen predictions from real runs and assert the published numbers still fall
out. That catches the failure mode that actually bites: someone adjusts the
scorer, the validation rules, or the aggregation, and every headline number
moves without anyone noticing.

It deliberately does not catch model drift - that needs a GPU run. What it
guarantees is that the measuring instrument has not changed shape.
"""
import json
from pathlib import Path

import pytest

from distill.calibrate import ece, mce
from distill.schema import TARGET_FIELDS
from distill.score import score
from distill.validate import apply_validation

FIX = Path(__file__).parent / "fixtures"
ACCEPT = 0.999

# Published in the README. Tight tolerance: these should only move when someone
# means to move them, and then this file changes in the same commit.
TOL = 0.005


@pytest.fixture(scope="module")
def extraction():
    return json.loads((FIX / "baseline_extraction.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def confidence():
    return json.loads((FIX / "baseline_confidence.json").read_text(encoding="utf-8"))["rows"]


# ---------- headline extraction numbers ----------

def test_overall_extraction_f1_holds(extraction):
    r = score([x["pred"] for x in extraction["rows"]],
              [x["gold"] for x in extraction["rows"]],
              fields=["store_name", "date", "subtotal_price", "tax_price", "total_price"])
    assert r["overall"]["f1"] == pytest.approx(0.804, abs=TOL)


@pytest.mark.parametrize("field,f1", [
    ("store_name", 0.820), ("date", 0.802), ("subtotal_price", 0.747),
    ("tax_price", 0.852), ("total_price", 0.800),
])
def test_per_field_f1_holds(extraction, field, f1):
    r = score([x["pred"] for x in extraction["rows"]],
              [x["gold"] for x in extraction["rows"]], fields=[field])
    assert r["per_field"][field]["f1"] == pytest.approx(f1, abs=TOL)


# ---------- confidence gating ----------

def _slices(rows):
    acc = [d for r in rows for d in r["fields"].values() if d["confidence"] >= ACCEPT]
    flag = [d for r in rows for d in r["fields"].values() if d["confidence"] < ACCEPT]
    return acc, flag


def test_auto_accept_slice_accuracy_holds(confidence):
    acc, _ = _slices(confidence)
    assert sum(d["correct"] for d in acc) / len(acc) == pytest.approx(0.944, abs=TOL)


def test_flagged_slice_accuracy_holds(confidence):
    _, flag = _slices(confidence)
    assert sum(d["correct"] for d in flag) / len(flag) == pytest.approx(0.541, abs=TOL)


def test_coverage_holds(confidence):
    acc, flag = _slices(confidence)
    assert len(acc) / (len(acc) + len(flag)) == pytest.approx(0.593, abs=TOL)


def test_separation_stays_wide(confidence):
    """The product thesis: trusted fields must be far better than flagged ones."""
    acc, flag = _slices(confidence)
    a = sum(d["correct"] for d in acc) / len(acc)
    f = sum(d["correct"] for d in flag) / len(flag)
    assert a - f > 0.35


def test_end_to_end_delivered_accuracy_holds(confidence):
    """Auto-accepted as-is + a human fixing everything flagged."""
    acc, flag = _slices(confidence)
    delivered = (sum(d["correct"] for d in acc) + len(flag)) / (len(acc) + len(flag))
    assert delivered == pytest.approx(0.967, abs=TOL)


# ---------- calibration ----------

def test_calibration_error_holds(confidence):
    assert ece(confidence) == pytest.approx(0.115, abs=TOL)
    assert mce(confidence) == pytest.approx(0.275, abs=TOL)


# ---------- validation behaviour on real rows ----------

def test_validation_still_flags_the_costco_pattern(confidence):
    """A subtotal equal to the total with no tax printed must stay flagged.

    This is the case the README screenshots; if a validation change stopped
    catching it, the demo would quietly become a lie.
    """
    f = {"subtotal_price": {"value": "70.34", "confidence": 1.0},
         "tax_price": {"value": None, "confidence": 1.0},
         "tip_price": {"value": None, "confidence": 1.0},
         "total_price": {"value": "70.34", "confidence": 1.0}}
    out = apply_validation(f)
    assert out["subtotal_price"]["confidence"] <= 0.5
    assert out["subtotal_price"]["validation_flag"] == "arithmetic"


def test_fixtures_cover_every_target_field(confidence):
    seen = {k for r in confidence for k in r["fields"]}
    assert set(TARGET_FIELDS) - {"tip_price"} <= seen
