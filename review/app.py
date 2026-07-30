"""Distill review queue - FastAPI backend.

Serves two kinds of receipt:
  - eval receipts (integer ids) replayed from a saved confidence run, which
    carry gold labels so accuracy stats are meaningful
  - uploaded receipts (ids like "u0"), extracted live by the model; no gold
    labels exist for these, so they show confidence but no accuracy

Both flow through the same confidence gate: fields where all k samples agreed
and validation passed are auto-accepted, the rest are flagged for a human.
Corrections persist to outputs/reviews.json.

Run:  uvicorn review.app:app --port 8990
"""
import io
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "wildreceipt"
RESULTS = ROOT / "outputs" / "confidence_results_v3.json"
REVIEWS = ROOT / "outputs" / "reviews.json"
UPLOAD_DIR = ROOT / "outputs" / "uploads"
UPLOAD_INDEX = UPLOAD_DIR / "index.json"
ACCEPT = 0.999   # all k samples agreed and validation passed

FIELD_ORDER = ["store_name", "date", "subtotal_price", "tax_price",
               "tip_price", "total_price"]
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Distill review queue")

with open(RESULTS, encoding="utf-8") as fh:
    ROWS = json.load(fh)["rows"]

# Importing torch and loading the VLM costs ~60s. Doing it on the first upload
# makes that request look hung, so warm it in the background at startup: the
# queue is served immediately while the model loads behind it.
MODEL_STATE = {"ready": False, "error": None}


def _warm():
    try:
        from distill.extract import _load
        _load()
        MODEL_STATE["ready"] = True
    except Exception as exc:                      # keep serving the eval queue
        MODEL_STATE["error"] = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def _startup():
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/api/status")
def status():
    return MODEL_STATE


def _load_uploads() -> list:
    if UPLOAD_INDEX.exists():
        return json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
    return []


UPLOADS = _load_uploads()


def _save_uploads():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_INDEX.write_text(json.dumps(UPLOADS, indent=2), encoding="utf-8")


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
                    "accepted": d["confidence"] >= ACCEPT,
                    "flag": d.get("validation_flag")})
    return out


def _find(rid: str):
    """Resolve a receipt id to (row, is_upload). Uploads use 'u<n>' ids."""
    if rid.startswith("u"):
        i = int(rid[1:]) if rid[1:].isdigit() else -1
        if not 0 <= i < len(UPLOADS):
            raise HTTPException(404)
        return UPLOADS[i], True
    if not rid.isdigit() or not 0 <= int(rid) < len(ROWS):
        raise HTTPException(404)
    return ROWS[int(rid)], False


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
                accept_ok += int(row["fields"][f["name"]].get("correct", False))
        store = row["fields"].get("store_name", {}).get("value") or "UNKNOWN"
        receipts.append({"id": str(i), "store": store, "flagged": len(flagged),
                         "total": len(fields), "reviewed": str(i) in reviews,
                         "uploaded": False})

    uploads = []
    for i, row in enumerate(UPLOADS):
        rid = f"u{i}"
        fields = _fields(row)
        store = row["fields"].get("store_name", {}).get("value") or "UNTITLED"
        uploads.append({"id": rid, "store": store,
                        "flagged": len([f for f in fields if not f["accepted"]]),
                        "total": len(fields), "reviewed": rid in reviews,
                        "uploaded": True})

    return {
        "receipts": receipts,
        "uploads": uploads,
        "stats": {
            "auto_accept_rate": round(accept_n / total_n, 3) if total_n else 0,
            "accept_accuracy": round(accept_ok / accept_n, 3) if accept_n else 0,
            "n_receipts": len(ROWS),
        },
    }


@app.get("/api/receipt/{rid}")
def receipt(rid: str):
    row, is_upload = _find(rid)
    items = row.get("line_items")
    if isinstance(items, dict):
        items = items.get("pred")
    return {"id": rid, "fields": _fields(row), "line_items": items,
            "review": _reviews().get(rid), "uploaded": is_upload,
            "elapsed": row.get("elapsed")}


@app.get("/api/image/{rid}")
def image(rid: str):
    row, is_upload = _find(rid)
    base = UPLOAD_DIR if is_upload else IMAGES
    path = (base / row["file"]).resolve()
    if not path.is_file() or base.resolve() not in path.parents:
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/api/extract")
def extract_upload(file: UploadFile = File(...), k: int = 3, items: bool = False):
    """Run the full pipeline on an uploaded receipt.

    Deliberately a sync def: FastAPI runs it in a threadpool, so the blocking
    GPU call does not stall the event loop. k defaults to 3 (not the eval
    default of 5) to keep interactive latency tolerable while still producing
    a real agreement-based confidence signal.
    """
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"unsupported type {file.content_type}")
    blob = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 12MB)")
    try:
        img = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")

    # Imported lazily so the server starts without touching CUDA.
    from distill.extract import extract_samples
    from distill.schema import TARGET_FIELDS
    from distill.score import aggregate
    from distill.validate import apply_validation
    from run_confidence import vote_line_items

    t0 = time.time()
    preds = extract_samples(img, k=max(1, min(k, 5)), items=items)
    line_items, _sum, iconf = vote_line_items(preds) if items else ([], None, 0)
    fields = apply_validation(aggregate(preds, TARGET_FIELDS),
                              line_items=line_items if iconf >= 0.6 else None)
    elapsed = round(time.time() - t0, 1)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.jpg"
    img.save(UPLOAD_DIR / name, "JPEG", quality=90)

    row = {"file": name, "fields": fields, "elapsed": elapsed}
    if items:
        row["line_items"] = line_items
    UPLOADS.append(row)
    _save_uploads()
    return {"id": f"u{len(UPLOADS) - 1}", "elapsed": elapsed,
            "fields": _fields(row), "line_items": line_items or None}


class Review(BaseModel):
    corrections: dict


@app.post("/api/receipt/{rid}/review")
def save_review(rid: str, body: Review):
    _find(rid)
    reviews = _reviews()
    reviews[rid] = {"corrections": body.corrections}
    REVIEWS.parent.mkdir(exist_ok=True)
    REVIEWS.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
