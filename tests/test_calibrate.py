"""Tests for calibration metrics.

Fixtures mirror the shape of a confidence run: rows of fields carrying a
confidence and whether that field turned out correct.
"""
import pytest

from distill.calibrate import bins, ece, mce, fit, apply_fit


def rows_from(spec):
    """spec: [(confidence, n_total, n_correct)] -> confidence-run shaped rows."""
    out = []
    for i, (conf, n, ok) in enumerate(spec):
        for j in range(n):
            out.append({"fields": {f"f{i}_{j}": {
                "value": "x", "confidence": conf, "correct": j < ok}}})
    return out


def test_perfect_calibration_is_zero_error():
    # 80% of fields at 0.8 confidence are correct; likewise 50% at 0.5
    r = rows_from([(0.8, 10, 8), (0.5, 10, 5)])
    assert ece(r) == 0.0
    assert mce(r) == 0.0


def test_overconfidence_is_measured():
    # claims 1.0, delivers 0.5
    r = rows_from([(1.0, 10, 5)])
    assert ece(r) == pytest.approx(0.5)
    assert mce(r) == pytest.approx(0.5)


def test_underconfidence_counts_too():
    # calibration error is symmetric: being pessimistic is also miscalibrated
    r = rows_from([(0.5, 10, 9)])
    assert ece(r) == pytest.approx(0.4)


def test_ece_is_weighted_by_bin_size():
    # a large well-calibrated bin should dominate a tiny broken one
    r = rows_from([(1.0, 99, 99), (0.2, 1, 1)])
    assert ece(r) == pytest.approx(0.008, abs=1e-3)
    assert mce(r) == pytest.approx(0.8)      # MCE ignores size


def test_bins_are_sorted_and_complete():
    b = bins(rows_from([(0.5, 4, 2), (1.0, 6, 6)]))
    assert [x["confidence"] for x in b] == [1.0, 0.5]
    assert [x["n"] for x in b] == [6, 4]
    assert b[0]["accuracy"] == 1.0 and b[1]["accuracy"] == 0.5


def test_empty_input_does_not_divide_by_zero():
    assert ece([]) == 0.0
    assert mce([]) == 0.0


# ---------- fitting ----------

def test_fit_maps_stated_to_observed():
    m = fit(rows_from([(1.0, 10, 9), (0.5, 10, 7)]))
    assert m[1.0] == pytest.approx(0.9)
    assert m[0.5] == pytest.approx(0.7)


def test_fit_skips_levels_with_too_little_data():
    # a level seen twice should not earn a correction
    m = fit(rows_from([(1.0, 10, 9), (0.3, 2, 0)]), min_n=5)
    assert 1.0 in m and 0.3 not in m


def test_apply_fit_falls_back_to_nearest_level():
    m = {1.0: 0.94, 0.5: 0.71}
    assert apply_fit(m, 1.0) == 0.94
    assert apply_fit(m, 0.55) == 0.71      # nearest known level
    assert apply_fit(m, 0.9) == 0.94


def test_apply_fit_without_a_mapping_is_identity():
    assert apply_fit({}, 0.6) == 0.6


def test_recalibration_reduces_in_sample_error():
    r = rows_from([(1.0, 20, 12), (0.5, 20, 16)])
    before = ece(r)
    m = fit(r)
    recal = [{"fields": {k: {**d, "confidence": apply_fit(m, d["confidence"])}
                         for k, d in row["fields"].items()}} for row in r]
    assert ece(recal) < before
    assert ece(recal) == pytest.approx(0.0, abs=1e-3)
