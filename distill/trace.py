"""Per-document tracing: latency, work done, and what it would have cost.

Local inference is free, which makes "cost per document" easy to wave away.
It is still worth measuring, for two reasons: latency decides whether a
reviewer waits on the tool, and the counterfactual cost of the same volume on
a hosted API is what justifies running locally at all.

Events are appended as JSONL - one line per extraction, no database, readable
with any text tool and cheap to ship alongside the outputs it describes.
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path("outputs/traces.jsonl")

# Rough public per-image pricing for a hosted vision model, used only to state
# the counterfactual. Adjust to whatever you would actually have bought.
HOSTED_USD_PER_DOC = 0.004


def record(event: dict, path: Path = DEFAULT_LOG) -> dict:
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


@contextmanager
def traced(op: str, path: Path = DEFAULT_LOG, **fields):
    """Time a block and record it, including when it raises.

    A failed extraction that is never logged makes the latency stats look
    better than the system behaves, so failures are recorded too.
    """
    started = time.time()
    info: dict = {}
    try:
        yield info
    except Exception as exc:
        record({"op": op, "seconds": round(time.time() - started, 3),
                "ok": False, "error": type(exc).__name__, **fields, **info}, path)
        raise
    record({"op": op, "seconds": round(time.time() - started, 3),
            "ok": True, **fields, **info}, path)


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(int(round(q * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[i]


def summary(path: Path = DEFAULT_LOG, op: str | None = None) -> dict:
    if not Path(path).exists():
        return {"n": 0}
    events = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:      # a partial final line should not
            continue                       # take down the whole report
        if op is None or e.get("op") == op:
            events.append(e)
    if not events:
        return {"n": 0}

    ok = [e for e in events if e.get("ok", True)]
    secs = sorted(e["seconds"] for e in ok if "seconds" in e)
    total = sum(secs)
    return {
        "n": len(events),
        "failures": len(events) - len(ok),
        "p50_seconds": _pct(secs, 0.50),
        "p95_seconds": _pct(secs, 0.95),
        "max_seconds": secs[-1] if secs else None,
        "total_seconds": round(total, 1),
        "docs_per_hour": round(3600 / (total / len(secs)), 1) if secs else None,
        "usd_local": 0.0,
        "usd_if_hosted": round(len(ok) * HOSTED_USD_PER_DOC, 4),
    }
