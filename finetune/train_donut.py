"""Fine-tune Donut (naver-clova-ix/donut-base) on WildReceipt receipts.

Conservative settings for a 12GB GPU: reduced resolution, gradient
checkpointing, bf16, small batch with gradient accumulation.

Two things make a long run informative rather than hopeful:

  * **Validation loss.** Held out from `test.jsonl` beyond the slice the
    reported evaluation uses, so it is never trained on and never leaks into
    published numbers. Without it there is no way to tell whether another
    fifty epochs would help or just memorise the training set.
  * **A cosine schedule with warmup**, the recipe the Donut paper uses and
    which earlier runs here skipped. A constant rate lets the model bounce
    around a minimum instead of settling into it.

Smoke (one loop, verify it runs and fits VRAM):
    python -u -m finetune.train_donut --n_train 16 --accum 2 --max_steps 3
Full:
    python -u -m finetune.train_donut --epochs 60
"""
import argparse
import json
import math
import time
from pathlib import Path

import random

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel

from finetune.donut_format import special_tokens, TASK_START

BASE = Path("data/wildreceipt")
MODEL = "naver-clova-ix/donut-base"
EVAL_SLICE = 100        # test.jsonl[:EVAL_SLICE] is the reported eval set


def augment(img, rng):
    """Light photometric and geometric jitter.

    1,112 training receipts is little enough that the model memorises them:
    training loss reached 0.42 while held-out validation sat at 1.23. These are
    the distortions a phone camera actually introduces - slight rotation, uneven
    exposure, a border of desk around the paper - so they widen the data without
    inventing receipts that could not exist.
    """
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-2.0, 2.0), resample=Image.BILINEAR,
                         expand=True, fillcolor=(255, 255, 255))
    if rng.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.20))
    if rng.random() < 0.5:
        w, h = img.size
        pad = [int(rng.uniform(0, 0.04) * w) for _ in range(2)]
        canvas = Image.new("RGB", (w + pad[0] + pad[1], h), (255, 255, 255))
        canvas.paste(img, (pad[0], 0))
        img = canvas
    return img


class ReceiptDS(Dataset):
    def __init__(self, rows, processor, max_len, augment_data=False):
        self.rows = rows
        self.p = processor
        self.max_len = max_len
        self.augment_data = augment_data

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(BASE / r["image"]).convert("RGB")
        if self.augment_data:
            # seeded per (index, epoch-ish) so workers do not share a stream
            img = augment(img, random.Random(random.randrange(1 << 30)))
        pv = self.p(img, return_tensors="pt").pixel_values[0]
        tok = self.p.tokenizer(
            r["target"] + self.p.tokenizer.eos_token,
            add_special_tokens=False, max_length=self.max_len + 1,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ids = tok.input_ids[0]
        # Shift explicitly rather than letting the model derive decoder inputs
        # from labels: the target already begins with <s_receipt>, and
        # decoder_start_token_id is that same token, so the automatic shift
        # prepends a second one and teaches a misaligned mapping.
        labels = ids[1:].clone()
        labels[labels == self.p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "decoder_input_ids": ids[:-1].clone(),
                "labels": labels}


def _rows(path, start=0, end=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows[start:end]


def _save(model, processor, path):
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"saved to {path}", flush=True)


def _schedule(opt, total, warmup_frac):
    """Linear warmup, then cosine decay to ~0."""
    warm = max(1, int(total * warmup_frac))

    def fn(step):
        if step < warm:
            return step / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    total, n = 0.0, 0
    for batch in dl:
        out = model(pixel_values=batch["pixel_values"].to(device, torch.bfloat16),
                    decoder_input_ids=batch["decoder_input_ids"].to(device),
                    labels=batch["labels"].to(device))
        total += out.loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--img_h", type=int, default=1280)
    ap.add_argument("--img_w", type=int, default=960)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="disable train-time image jitter")
    ap.add_argument("--val_n", type=int, default=80,
                    help="validation rows from test.jsonl beyond the eval slice")
    ap.add_argument("--patience", type=int, default=10,
                    help="epochs without validation improvement before stopping")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--save_dir", default="finetune/donut-distill")
    args = ap.parse_args()

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

    train_rows = _rows("finetune/data/train.jsonl", 0, args.n_train)
    val_rows = _rows("finetune/data/test.jsonl", EVAL_SLICE, EVAL_SLICE + args.val_n)
    tr = DataLoader(ReceiptDS(train_rows, processor, args.max_len,
                              augment_data=args.augment),
                    batch_size=args.batch, shuffle=True, num_workers=2)
    va = DataLoader(ReceiptDS(val_rows, processor, args.max_len),
                    batch_size=args.batch, num_workers=2)

    steps_per_epoch = max(1, len(tr) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = _schedule(opt, total_steps, args.warmup)

    print(f"train={len(train_rows)} val={len(val_rows)}  img={args.img_h}x{args.img_w}  "
          f"batch={args.batch} accum={args.accum} lr={args.lr} "
          f"warmup={args.warmup} steps={total_steps}", flush=True)

    step, best, since_best = 0, float("inf"), 0
    t0 = time.time()
    for epoch in range(args.epochs):
        run, seen = 0.0, 0
        for bi, batch in enumerate(tr):
            out = model(pixel_values=batch["pixel_values"].to(device, torch.bfloat16),
                        decoder_input_ids=batch["decoder_input_ids"].to(device),
                        labels=batch["labels"].to(device))
            (out.loss / args.accum).backward()
            run += out.loss.item()
            seen += 1
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"  epoch {epoch} step {step}/{total_steps} "
                          f"loss {run / seen:.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"t {time.time() - t0:.0f}s", flush=True)
                    run, seen = 0.0, 0
                if args.max_steps and step >= args.max_steps:
                    print("max_steps reached (smoke ok).", flush=True)
                    return

        vloss = validate(model, va, device)
        mark = ""
        if vloss < best - 1e-4:
            best, since_best = vloss, 0
            _save(model, processor, f"{args.save_dir}-best")
            mark = "  <- best"
        else:
            since_best += 1
        print(f"epoch {epoch + 1}/{args.epochs} val_loss {vloss:.4f} "
              f"(best {best:.4f}, {since_best} since){mark}", flush=True)

        if (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            _save(model, processor, f"{args.save_dir}-ep{epoch + 1}")
        if since_best >= args.patience:
            print(f"no validation improvement in {args.patience} epochs; stopping.",
                  flush=True)
            break

    _save(model, processor, args.save_dir)
    print(f"best validation loss {best:.4f}; best weights in {args.save_dir}-best",
          flush=True)


if __name__ == "__main__":
    main()
