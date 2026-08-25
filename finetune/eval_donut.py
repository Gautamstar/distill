"""Evaluate a fine-tuned Donut checkpoint on the same harness as the zero-shot
VLM, so the two numbers are directly comparable.

Reuses distill.score, so field matching (price normalization, fuzzy text) is
identical to the baseline in run_phase0.py. Reports field-level P/R/F1 plus
line-item F1 and per-document latency.

Run:  python -m finetune.eval_donut --model finetune/donut-distill --n 100
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from distill.schema import TARGET_FIELDS
from distill.score import score, score_line_items
from finetune.donut_format import from_sequence, TASK_START

BASE = Path("data/wildreceipt")


def load(path):
    processor = DonutProcessor.from_pretrained(path)
    model = VisionEncoderDecoderModel.from_pretrained(path)
    model.to("cuda", dtype=torch.bfloat16).eval()
    return processor, model


@torch.no_grad()
def predict(processor, model, image):
    pv = processor(image, return_tensors="pt").pixel_values.to("cuda", torch.bfloat16)
    start = processor.tokenizer(TASK_START, add_special_tokens=False,
                                return_tensors="pt").input_ids.to("cuda")
    tok = processor.tokenizer
    kw = {}
    if tok.unk_token_id is not None:      # absent on some tokenizer configs
        kw["bad_words_ids"] = [[tok.unk_token_id]]
    out = model.generate(
        pv, decoder_input_ids=start, max_length=768,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        use_cache=True, return_dict_in_generate=True, **kw,
    )
    seq = processor.batch_decode(out.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, "")
    return from_sequence(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="finetune/donut-distill")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="outputs/donut_eval.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("finetune/data/test.jsonl", encoding="utf-8")]
    rows = rows[:args.n]
    processor, model = load(args.model)

    preds, golds, rows_out = [], [], []
    item_agg = {"tp": 0, "fp": 0, "fn": 0}
    t0 = time.time()
    for i, r in enumerate(rows):
        img = Image.open(BASE / r["image"]).convert("RGB")
        p = predict(processor, model, img)
        g = from_sequence(r["target"])
        preds.append(p)
        golds.append(g)
        s = score_line_items(p.get("line_items") or [], g.get("line_items") or [])
        for k in ("tp", "fp", "fn"):
            item_agg[k] += s[k]
        rows_out.append({"image": r["image"], "pred": p, "gold": g})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    elapsed = time.time() - t0

    results = score(preds, golds, TARGET_FIELDS)
    ip = item_agg["tp"] / max(item_agg["tp"] + item_agg["fp"], 1)
    ir = item_agg["tp"] / max(item_agg["tp"] + item_agg["fn"], 1)
    if1 = 2 * ip * ir / (ip + ir) if (ip + ir) else 0

    print("\n============ FINE-TUNED DONUT ============")
    for f in TARGET_FIELDS:
        m = results["per_field"][f]
        print(f"{f:16s} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    o = results["overall"]
    print(f"{'OVERALL':16s} F1={o['f1']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}")
    print(f"{'line items':16s} F1={if1:.3f}  P={ip:.3f}  R={ir:.3f}")
    print(f"docs={len(rows)}  {elapsed / len(rows):.2f}s/doc")
    print("==========================================")

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "line_items": {"f1": if1, **item_agg},
                   "sec_per_doc": elapsed / len(rows), "rows": rows_out},
                  fh, indent=2, ensure_ascii=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
