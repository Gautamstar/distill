"""Fine-tune Donut (naver-clova-ix/donut-base) on WildReceipt receipts.

Conservative settings for a 12GB GPU: reduced resolution, gradient
checkpointing, bf16, batch 1 with gradient accumulation.

Smoke (one loop, verify it runs + fits VRAM):
    python -u -m finetune.train_donut --n_train 8 --accum 2 --max_steps 3
Full:
    python -u -m finetune.train_donut --epochs 3 --save_dir finetune/donut-distill
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel

from finetune.donut_format import special_tokens, TASK_START

BASE = Path("data/wildreceipt")
MODEL = "naver-clova-ix/donut-base"


class ReceiptDS(Dataset):
    def __init__(self, path, processor, max_len, n=None):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if n:
            self.rows = self.rows[:n]
        self.p = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(BASE / r["image"]).convert("RGB")
        pv = self.p(img, return_tensors="pt").pixel_values[0]
        tok = self.p.tokenizer(
            r["target"] + self.p.tokenizer.eos_token,
            add_special_tokens=False, max_length=self.max_len + 1,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ids = tok.input_ids[0]
        # Shift explicitly instead of letting the model derive decoder inputs
        # from labels: the target already begins with <s_receipt>, and
        # decoder_start_token_id is that same token, so the automatic shift
        # prepends a second one and teaches a misaligned mapping.
        decoder_input_ids = ids[:-1].clone()
        labels = ids[1:].clone()
        labels[labels == self.p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "decoder_input_ids": decoder_input_ids,
                "labels": labels}


def _save(model, processor, path):
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"saved to {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--img_h", type=int, default=1280)
    ap.add_argument("--img_w", type=int, default=960)
    ap.add_argument("--max_len", type=int, default=768)
    # batch 2 peaks near ~7GB on a 12GB card (batch 1 measured 4.0GB), which
    # leaves room for desktop apps; accum keeps the effective batch at 8.
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--resume", default=None,
                    help="continue from a checkpoint dir instead of donut-base")
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--save_dir", default="finetune/donut-distill")
    args = ap.parse_args()

    # --resume continues from a checkpoint (tokenizer and resized embeddings
    # are already saved there) instead of restarting from donut-base.
    src = args.resume or MODEL
    processor = DonutProcessor.from_pretrained(src)
    processor.image_processor.size = {"height": args.img_h, "width": args.img_w}
    processor.image_processor.do_align_long_axis = False
    if not args.resume:
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens()})

    model = VisionEncoderDecoderModel.from_pretrained(src)
    if not args.resume:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.encoder.image_size = [args.img_h, args.img_w]
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_START)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    device = "cuda"
    model.to(device, dtype=torch.bfloat16)
    model.train()

    ds = ReceiptDS("finetune/data/train.jsonl", processor, args.max_len, args.n_train)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"train={len(ds)}  img={args.img_h}x{args.img_w}  "
          f"batch={args.batch} accum={args.accum}  lr={args.lr}", flush=True)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        run, seen = 0.0, 0
        for bi, batch in enumerate(dl):
            pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            labels = batch["labels"].to(device)
            dii = batch["decoder_input_ids"].to(device)
            out = model(pixel_values=pv, decoder_input_ids=dii, labels=labels)
            (out.loss / args.accum).backward()
            run += out.loss.item()
            seen += 1
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"epoch {epoch} step {step} loss {run / seen:.4f} "
                          f"peakVRAM {mem:.1f}GB t {time.time() - t0:.0f}s", flush=True)
                    run, seen = 0.0, 0
                if args.max_steps and step >= args.max_steps:
                    print("max_steps reached (smoke ok).", flush=True)
                    return
        # periodic checkpoints so a long run survives a crash and an earlier
        # epoch stays available if later ones overfit; every epoch would be
        # ~800MB each, so space them out
        if (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            _save(model, processor, f"{args.save_dir}-ep{epoch + 1}")
    _save(model, processor, args.save_dir)


if __name__ == "__main__":
    main()
