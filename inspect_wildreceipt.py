"""Inspect the locally-downloaded WildReceipt data (from the MMOCR tarball)
so we can map its label categories to Distill's schema.

Prereq (download + extract):
  mkdir -p data && cd data
  wget https://download.openmmlab.com/mmocr/data/wildreceipt.tar
  tar -xf wildreceipt.tar
  cd ..

Run:  python inspect_wildreceipt.py
"""
import json
from pathlib import Path

BASE = Path("data/wildreceipt")


def main():
    class_list = BASE / "class_list.txt"
    train = BASE / "train.txt"
    if not class_list.exists():
        raise SystemExit(f"Not found: {class_list}. Did the download/extract step run?")

    print("===== class_list.txt (label id -> name) =====")
    id2name = {}
    for line in class_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        idx = int(parts[0])
        name = parts[1] if len(parts) > 1 else ""
        id2name[idx] = name
        print(f"  {idx}: {name}")

    print("\n===== first annotation line of train.txt =====")
    with open(train, encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    print("file_name:", rec.get("file_name"))
    img_path = BASE / rec["file_name"]
    print("image exists:", img_path.exists())
    try:
        from PIL import Image
        im = Image.open(img_path)
        print("image opened:", im.size, im.mode)
    except Exception as e:
        print("image open failed:", repr(e))

    print("\nlabel -> text pairs (first 40 boxes):")
    for ann in rec.get("annotations", [])[:40]:
        lbl = ann.get("label")
        print(f"  {id2name.get(lbl, lbl)!r:26s} <- {ann.get('text')!r}")


if __name__ == "__main__":
    main()
