"""Print the raw sequence a Donut checkpoint generates, before parsing.

An all-null evaluation can mean the model is undertrained or that decoding /
parsing is broken; only the raw string distinguishes them.

Run:  python -m finetune.debug_generate --model finetune/donut-distill --n 2
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from finetune.donut_format import TASK_START, from_sequence

BASE = Path("data/wildreceipt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="finetune/donut-distill")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    processor = DonutProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.to("cuda", dtype=torch.bfloat16).eval()
    tok = processor.tokenizer

    print("start token:", TASK_START, "->", tok.convert_tokens_to_ids(TASK_START))
    print("eos:", tok.eos_token, tok.eos_token_id, "| pad:", tok.pad_token, tok.pad_token_id)
    print("decoder_start_token_id:", model.config.decoder_start_token_id)
    print("image size:", processor.image_processor.size)
    print("vocab:", len(tok), "| embedding rows:",
          model.decoder.get_input_embeddings().weight.shape[0])

    rows = [json.loads(l) for l in open("finetune/data/test.jsonl", encoding="utf-8")][:args.n]
    for r in rows:
        img = Image.open(BASE / r["image"]).convert("RGB")
        pv = processor(img, return_tensors="pt").pixel_values.to("cuda", torch.bfloat16)
        start = tok(TASK_START, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(pv, decoder_input_ids=start, max_length=256,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id, use_cache=True)
        print("\n=== ", r["image"].split("/")[-1][:16])
        print("ids[:20]:", out[0][:20].tolist())
        print("RAW   :", repr(tok.decode(out[0], skip_special_tokens=False))[:420])
        print("PARSED:", from_sequence(tok.decode(out[0], skip_special_tokens=False)))
        print("TARGET:", r["target"][:160])


if __name__ == "__main__":
    main()
