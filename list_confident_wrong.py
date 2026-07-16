"""List fields the model was fully confident about (all k samples agreed) but
got wrong. These are the dangerous cases for auto-accept: self-consistency
cannot catch them, so they bound the accuracy of the confident slice.

Run:  python list_confident_wrong.py
"""
import json


def main(path: str = "outputs/confidence_results.json"):
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    cases = []
    for r in rows:
        for f, d in r["fields"].items():
            if d["confidence"] >= 0.999 and not d["correct"]:
                cases.append((r["file"], f, d["value"], d["gold"]))
    print(f"{len(cases)} confident-but-wrong fields:\n")
    for file, f, val, gold in cases:
        tag = "FALSE-POSITIVE" if gold is None else ("MISSED" if val is None else "WRONG")
        print(f"[{tag:14s}] {f:16s} pred={val!r:22} gold={gold!r}")
        print(f"                 {file}")


if __name__ == "__main__":
    main()
