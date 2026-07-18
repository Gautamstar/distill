"""Distill review queue - FastAPI backend.

Serves the confidence-gated extraction results as a human review queue: the
model auto-accepts fields where all k samples agreed and validation passed;
only the flagged fields need a human. Corrections are persisted to
outputs/reviews.json.

Run:  uvicorn review.app:app --port 8990
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "wildreceipt"
RESULTS = ROOT / "outputs" / "confidence_results_v3.json"
REVIEWS = ROOT / "outputs" / "reviews.json"
ACCEPT = 0.999   # all k samples agreed and validation passed

FIELD_ORDER = ["store_name", "date", "subtotal_price", "tax_price",
               "tip_price", "total_price"]

app = FastAPI(title="Distill review queue")

with open(RESULTS, encoding="utf-8") as fh:
    ROWS = json.load(fh)["rows"]


def _reviews() -> dict:
    if REVIEWS.exists():
        return json.loads(REVIEWS.read_text(encoding="utf-8"))
    return {}


def _fields(row):
    out = []
    for f in FIELD_ORDER:
        d = row["fields"].get(f)
        if d is None:
            continue
        out.append({"name": f, "value": d["value"],
                    "confidence": d["confidence"],
                    "accepted": d["confidence"] >= ACCEPT})
    return out


@app.get("/api/queue")
def queue():
    reviews = _reviews()
    receipts, accept_n, accept_ok, total_n = [], 0, 0, 0
    for i, row in enumerate(ROWS):
        fields = _fields(row)
        flagged = [f for f in fields if not f["accepted"]]
        for f in fields:
            total_n += 1
            if f["accepted"]:
                accept_n += 1
                d = row["fields"][f["name"]]
                accept_ok += int(d.get("correct", False))
        store = row["fields"].get("store_name", {}).get("value") or "UNKNOWN"
        receipts.append({"id": i, "store": store, "flagged": len(flagged),
                         "total": len(fields), "reviewed": str(i) in reviews})
    return {
        "receipts": receipts,
        "stats": {
            "auto_accept_rate": round(accept_n / total_n, 3) if total_n else 0,
            "accept_accuracy": round(accept_ok / accept_n, 3) if accept_n else 0,
            "n_receipts": len(ROWS),
        },
    }


@app.get("/api/receipt/{i}")
def receipt(i: int):
    if not 0 <= i < len(ROWS):
        raise HTTPException(404)
    row = ROWS[i]
    review = _reviews().get(str(i))
    return {"id": i, "file": row["file"], "fields": _fields(row),
            "line_items": row.get("line_items", {}).get("pred") if isinstance(
                row.get("line_items"), dict) else None,
            "review": review}


@app.get("/api/image/{i}")
def image(i: int):
    if not 0 <= i < len(ROWS):
        raise HTTPException(404)
    path = (IMAGES / ROWS[i]["file"]).resolve()
    if not path.is_file() or IMAGES.resolve() not in path.parents:
        raise HTTPException(404)
    return FileResponse(path)


class Review(BaseModel):
    corrections: dict


@app.post("/api/receipt/{i}/review")
def save_review(i: int, body: Review):
    if not 0 <= i < len(ROWS):
        raise HTTPException(404)
    reviews = _reviews()
    reviews[str(i)] = {"corrections": body.corrections}
    REVIEWS.parent.mkdir(exist_ok=True)
    REVIEWS.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
