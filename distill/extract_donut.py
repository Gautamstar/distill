"""Donut extraction backend, interface-compatible with `distill.extract`.

The zero-shot VLM is 3B parameters and needs a GPU, which is what makes hosting
expensive. A fine-tuned Donut is ~200M and runs on CPU, so it is the backend
that makes a free always-on deployment possible.

`extract_samples` matches the signature the review app and the confidence
harness already call, so swapping backends is configuration rather than a
rewrite. Confidence still comes from k-sample agreement: Donut generates
greedily by default, but sampling with a temperature gives the same
disagreement signal the VLM path relies on.
"""
import os
from functools import lru_cache

import torch

from .schema import TARGET_FIELDS

MODEL_DIR = os.environ.get("DISTILL_DONUT_DIR", "finetune/donut-distill-best")
# CPU is the point of this backend; CUDA is used only when explicitly available
# and not disabled, so the same image runs on a laptop and on a GPU box.
DEVICE = ("cuda" if torch.cuda.is_available()
          and os.environ.get("DISTILL_FORCE_CPU") != "1" else "cpu")


@lru_cache(maxsize=1)
def _load(model_dir: str = MODEL_DIR):
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    # bfloat16 on CPU is slower than float32 on most hardware, so only narrow
    # the dtype where it actually helps.
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model.to(DEVICE, dtype=dtype).eval()
    return processor, model


def _blank():
    out = {k: None for k in TARGET_FIELDS}
    out["line_items"] = []
    return out


@torch.no_grad()
def _generate(image, k: int, do_sample: bool, temperature: float = 0.7,
              items: bool = False, model_dir: str = MODEL_DIR):
    from .donut_runtime import parse_sequence, start_ids
    processor, model = _load(model_dir)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    pv = processor(image, return_tensors="pt").pixel_values.to(DEVICE, dtype)

    tok = processor.tokenizer
    kwargs = {"max_length": 512, "pad_token_id": tok.pad_token_id,
              "eos_token_id": tok.eos_token_id, "use_cache": True}
    if do_sample and k > 1:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9,
                      num_return_sequences=k)
    out = model.generate(pv, decoder_input_ids=start_ids(tok, DEVICE), **kwargs)
    texts = processor.batch_decode(out)
    return [parse_sequence(t, tok) for t in texts] or [_blank()]


def extract(image, model_dir: str = MODEL_DIR):
    """Single greedy extraction."""
    return _generate(image, k=1, do_sample=False, model_dir=model_dir)[0]


def extract_samples(image, k: int = 5, temperature: float = 0.7,
                    items: bool = False, model_dir: str = MODEL_DIR):
    """k extractions for self-consistency confidence, matching the VLM backend.

    `items` is accepted for interface parity. The fine-tune is trained on
    header fields only - line-item tags outnumbered header tags 5:1 and
    suppressed them - so it returns an empty item list rather than pretending.
    """
    if k <= 1:
        return [extract(image, model_dir)]
    return _generate(image, k=k, do_sample=True, temperature=temperature,
                     model_dir=model_dir)
