"""Capture README assets from the running review queue.

Scripted rather than hand-screenshotted so the images can be regenerated
whenever the UI changes. Produces:
  docs/demo.gif    the upload flow: drop a receipt, wait, fields appear
  docs/queue.png   the full review queue
  docs/caught.png  the money shot: a two-cent misread flagged by arithmetic

Prereq: the server is running and warm (uvicorn review.app:app --port 8990).

Run:  python capture_demo.py
"""
import json
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8990"
DOCS = Path("docs")
SHOTS = DOCS / "_frames"
VIEW = {"width": 1280, "height": 900}

# Receipts chosen for what they demonstrate, not at random.
#   CLEAN   Walmart: every field agrees and the totals add up, so all auto-accept
#   CAUGHT  Costco: no subtotal line is printed, the model copies the total, and
#           validation catches it. Chosen over the Cho Eun two-cent misread
#           because that one is a coin flip: when the model also computes a
#           matching total it is self-consistently wrong and arithmetic passes.
CLEAN = "data/wildreceipt/image_files/Image_59/20/96ae21a2a6cc19e60b28647bdb2d0603365bd0f0.jpeg"
MISREAD = "data/wildreceipt/image_files/Image_80/7/4381.jpeg"


def _settle(page, ms=450):
    page.wait_for_timeout(ms)


def main():
    DOCS.mkdir(exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    frames = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEW, device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        _settle(page, 900)

        def shot(name, hold=1):
            f = SHOTS / f"{name}.png"
            page.screenshot(path=str(f))
            for _ in range(hold):
                frames.append(f)
            return f

        # 1. resting state, before anything is uploaded
        shot("01_idle", hold=8)

        # 2. a clean receipt: everything agrees, everything auto-accepts
        page.set_input_files("#file", CLEAN)
        _settle(page, 700)
        shot("02_working", hold=6)
        page.wait_for_function(
            "document.getElementById('drop-note').textContent.includes('done')",
            timeout=120000)
        _settle(page, 800)
        shot("03_clean", hold=12)

        # 3. the one the README talks about: the model invents a subtotal and
        #    validation catches the incoherence.
        #
        #    Sampling is stochastic, so the same receipt does not always land on
        #    the arithmetic flag - sometimes the samples merely disagree, which
        #    illustrates a different (weaker) point. Retry until the run shows
        #    the behaviour the caption describes, so the asset cannot drift out
        #    of sync with the text. Every attempt is a real, unedited run.
        for attempt in range(4):
            page.set_input_files("#file", MISREAD)
            _settle(page, 700)
            if attempt == 0:
                shot("04_working2", hold=5)
            page.wait_for_function(
                "document.getElementById('drop-note').textContent.includes('done')",
                timeout=120000)
            _settle(page, 800)
            flags = page.evaluate(
                "[...document.querySelectorAll('.f-tag.rev')].map(e => e.textContent.trim())")
            if "arithmetic" in flags:
                break
            print(f"  attempt {attempt + 1}: flags={flags}, retrying", flush=True)

        # Retries leave near-duplicate rows in the queue. Keep the first upload
        # and the winning one; the server reads the index from disk per request,
        # so pruning here is picked up on reload.
        index = Path("outputs/uploads/index.json")
        rows = json.loads(index.read_text())
        if len(rows) > 2:
            keep = [rows[0], rows[-1]]
            for r in rows[1:-1]:
                (Path("outputs/uploads") / r["file"]).unlink(missing_ok=True)
            index.write_text(json.dumps(keep, indent=2))
            page.reload(wait_until="networkidle")
            _settle(page, 700)
            page.evaluate("openReceipt('u1')")
            _settle(page, 900)
        caught = shot("05_caught", hold=16)

        # Captured at 2x for sharpness, downscaled for the README: a 2560px
        # PNG of a dark UI is several MB and GitHub serves it on every view.
        def web(src, dest, width=1500):
            im = Image.open(src).convert("RGB")
            im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)\
              .save(DOCS / dest, quality=88, optimize=True)

        web(caught, "caught.jpg")
        page.screenshot(path=str(SHOTS / "queue_full.png"), full_page=True)
        web(SHOTS / "queue_full.png", "queue.jpg")
        browser.close()

    imgs = [Image.open(f).convert("RGB") for f in frames]
    w, h = imgs[0].size
    imgs = [im.resize((w // 3, h // 3), Image.LANCZOS) for im in imgs]
    imgs[0].save(DOCS / "demo.gif", save_all=True, append_images=imgs[1:],
                 duration=320, loop=0, optimize=True)

    for f in (DOCS / "demo.gif", DOCS / "caught.jpg", DOCS / "queue.jpg"):
        print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
