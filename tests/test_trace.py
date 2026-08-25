"""Tests for per-document tracing."""
import json

import pytest

from distill.trace import record, traced, summary, HOSTED_USD_PER_DOC


@pytest.fixture
def log(tmp_path):
    return tmp_path / "traces.jsonl"


def test_record_appends_one_line_per_event(log):
    record({"op": "extract", "seconds": 1.0, "ok": True}, log)
    record({"op": "extract", "seconds": 2.0, "ok": True}, log)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["seconds"] == 1.0


def test_record_stamps_a_timestamp(log):
    e = record({"op": "extract"}, log)
    assert e["ts"].endswith("+00:00") and "T" in e["ts"]


def test_traced_times_the_block_and_keeps_extra_fields(log):
    with traced("extract", log, k=3, doc="u1") as info:
        info["fields"] = 6
    e = json.loads(log.read_text(encoding="utf-8").strip())
    assert e["op"] == "extract" and e["ok"] is True
    assert e["k"] == 3 and e["doc"] == "u1" and e["fields"] == 6
    assert e["seconds"] >= 0


def test_traced_records_failures_and_reraises(log):
    with pytest.raises(ValueError):
        with traced("extract", log):
            raise ValueError("boom")
    e = json.loads(log.read_text(encoding="utf-8").strip())
    assert e["ok"] is False and e["error"] == "ValueError"


def test_failures_are_excluded_from_latency_but_counted(log):
    record({"op": "extract", "seconds": 1.0, "ok": True}, log)
    record({"op": "extract", "seconds": 99.0, "ok": False}, log)
    s = summary(log)
    assert s["n"] == 2 and s["failures"] == 1
    assert s["max_seconds"] == 1.0          # the failure must not skew latency


def test_percentiles_and_throughput(log):
    for v in [1.0, 2.0, 3.0, 4.0, 10.0]:
        record({"op": "extract", "seconds": v, "ok": True}, log)
    s = summary(log)
    assert s["p50_seconds"] == 3.0
    assert s["p95_seconds"] == 10.0
    assert s["docs_per_hour"] == pytest.approx(3600 / 4.0, abs=0.1)


def test_cost_states_the_counterfactual(log):
    for _ in range(10):
        record({"op": "extract", "seconds": 1.0, "ok": True}, log)
    s = summary(log)
    assert s["usd_local"] == 0.0
    assert s["usd_if_hosted"] == pytest.approx(10 * HOSTED_USD_PER_DOC)


def test_summary_filters_by_op(log):
    record({"op": "extract", "seconds": 1.0, "ok": True}, log)
    record({"op": "train", "seconds": 500.0, "ok": True}, log)
    assert summary(log, op="extract")["n"] == 1
    assert summary(log, op="extract")["max_seconds"] == 1.0


def test_missing_log_is_not_an_error(tmp_path):
    assert summary(tmp_path / "nope.jsonl") == {"n": 0}


def test_partial_final_line_does_not_break_the_report(log):
    record({"op": "extract", "seconds": 1.0, "ok": True}, log)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write('{"op": "extract", "seco')      # process died mid-write
    assert summary(log)["n"] == 1
