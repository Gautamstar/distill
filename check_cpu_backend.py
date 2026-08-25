"""Verify the Donut backend runs on CPU and report real CPU latency.

The deployment argument rests on this path working without a GPU, so it is
worth exercising directly rather than trusting the Dockerfile.

Run:  DISTILL_FORCE_CPU=1 python check_cpu_backend.py [checkpoint_dir]
"""
import sys
import time

from PIL import Image

CKPT = sys.argv[1] if len(sys.argv) > 1 else "finetune/donut-distill-best"
IMAGES = [
    "data/wildreceipt/image_files/Image_80/7/4381.jpeg",
    "data/wildreceipt/image_files/Image_59/20/96ae21a2a6cc19e60b28647bdb2d0603365bd0f0.jpeg",
    "data/wildreceipt/image_files/Image_12/10/845be0dd6f5b04866a2042abd28d558032ef2576.jpeg",
]


def main():
    from distill import extract_donut as backend
    print("checkpoint:", CKPT)
    print("device:", backend.DEVICE)

    t_load = time.time()
    backend._load(CKPT)
    print(f"model load: {time.time() - t_load:.1f}s")

    times = []
    for path in IMAGES:
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        rec = backend.extract(img, model_dir=CKPT)
        dt = time.time() - t0
        times.append(dt)
        fields = {k: v for k, v in rec.items() if k != "line_items" and v}
        print(f"  {dt:5.2f}s  {fields}")

    print(f"\nmean {sum(times) / len(times):.2f}s/doc on {backend.DEVICE}")


if __name__ == "__main__":
    main()
