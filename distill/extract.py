"""Zero-shot field extraction with Qwen2.5-VL-3B (local)."""
import json
import re
from functools import lru_cache

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .schema import ReceiptExtraction, TARGET_FIELDS

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "You are extracting fields from a store receipt image. "
    "Return ONLY a JSON object with exactly these keys: "
    "store_name, date, subtotal_price, tax_price, tip_price, total_price, "
    "line_items. "
    "store_name is the merchant or store name printed at the top. "
    "date is the transaction date as printed. "
    "For the money fields, read the amounts exactly as printed in the totals "
    "section; do not calculate or infer them. "
    "subtotal_price is the amount on a Subtotal or Sub Total line. "
    "tax_price is the amount on a Tax, Sales Tax, or VAT line. "
    "tip_price is the amount on a Tip or Gratuity line. "
    "total_price is the amount on the Total or Amount Due line. "
    "Use null for any money field whose labeled line is not printed. "
    "line_items is an array with one object per purchased product line, each "
    "with keys description, quantity, price (quantity null if not printed; "
    "price is the amount printed on that line). Do not include subtotal, tax, "
    "tip, total, change, or payment lines as items. "
    "Give numbers without the dollar sign (for example 12.99). "
    "Output no text other than the JSON."
)


@lru_cache(maxsize=1)
def _load():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _clean(v):
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in ("", "null", "none", "n/a", "na", "-") else s


def _run(image, do_sample: bool = False, temperature: float = 1.0) -> dict:
    model, processor = _load()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": 1024, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=0.9)
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    data = _parse_json(out_text)
    result = {k: _clean(data.get(k)) for k in TARGET_FIELDS}
    items = data.get("line_items")
    result["line_items"] = [
        {"description": _clean(i.get("description")),
         "quantity": _clean(i.get("quantity")),
         "price": _clean(i.get("price"))}
        for i in items if isinstance(i, dict)
    ] if isinstance(items, list) else []
    return result


def extract(image) -> ReceiptExtraction:
    """Single greedy extraction."""
    data = _run(image)
    items = data.pop("line_items", [])
    return ReceiptExtraction(**data, line_items=items)


def extract_samples(image, k: int = 5, temperature: float = 0.7):
    """k sampled extractions, for self-consistency confidence scoring.
    k=1 runs a single greedy pass (fast iteration mode: no confidence signal,
    every field reports 1.0)."""
    if k == 1:
        return [_run(image)]
    return [_run(image, do_sample=True, temperature=temperature) for _ in range(k)]
