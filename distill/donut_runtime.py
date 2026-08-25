"""Sequence handling shared by the Donut backend.

Kept apart from `finetune.donut_format` so serving does not import the training
package: a deployed container should carry the runtime, not the training code
and its dependencies.
"""
import re

from .schema import TARGET_FIELDS

TASK_START = "<s_receipt>"


def start_ids(tokenizer, device):
    ids = tokenizer(TASK_START, add_special_tokens=False,
                    return_tensors="pt").input_ids
    return ids.to(device)


def _value(seq: str, key: str):
    """Text between an opening tag and whatever ends it.

    Deliberately does not require the matching closing tag: the model routinely
    emits `<s_subtotal_price> 55.96</s_total_price>`, having identified the
    field and read the value correctly while picking the wrong closer.
    Requiring the exact pair throws away a correct extraction over a formatting
    artifact.
    """
    open_tag = f"<s_{key}>"
    exact = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(f"</s_{key}>"), seq)
    if exact:
        return exact.group(1).strip() or None
    loose = re.search(re.escape(open_tag) + r"\s*([^<]*)", seq)
    return (loose.group(1).strip() or None) if loose else None


def parse_sequence(seq: str, tokenizer=None) -> dict:
    """Generated sequence -> a record shaped like the VLM backend's output."""
    if tokenizer is not None:
        for special in (tokenizer.eos_token, tokenizer.pad_token):
            if special:
                seq = seq.replace(special, "")
    out = {k: _value(seq, k) for k in TARGET_FIELDS}
    out["line_items"] = []
    return out
