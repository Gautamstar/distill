"""Print the raw ground-truth structure of a few CORD receipts so we can
confirm the exact field keys before trusting the gold mapping in data.py.

Run:  python inspect_cord.py
"""
import json

from distill.data import _load_ds


def main(n: int = 3, split: str = "test"):
    ds = _load_ds(split).select(range(n))
    for i, ex in enumerate(ds):
        gt = json.loads(ex["ground_truth"])
        parse = gt.get("gt_parse", gt)
        print(f"\n===== Sample {i}: top-level keys = {list(parse.keys())} =====")
        print("total:", json.dumps(parse.get("total", {}), indent=2, ensure_ascii=False))
        print("sub_total:", json.dumps(parse.get("sub_total", parse.get("subtotal", {})),
                                       indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
