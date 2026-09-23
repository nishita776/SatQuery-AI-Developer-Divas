"""ML-07: RSVQA -> training/data/raw/rsvqa/ (images + records.jsonl).

The official RSVQA is on Zenodo; these are curated 2k subsets on the Hub with
the images embedded, which is why they are usable in minutes rather than hours.
Only a `validation` split exists, so we take all of it: ~4k against a 5k target.
"""

import argparse, json, sys
from pathlib import Path
from datasets import load_dataset

SOURCES = [("rsvqa-lr", "dmarsili/RSVQA-LR-2k"), ("rsvqa-hr", "dmarsili/RSVQA-HR-2k")]
OUT = Path("training/data/raw/rsvqa")

def main(limit_per_source):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for name, rid in SOURCES:
        print(f"\n=== {name}  ({rid}) ===", flush=True)
        ds = load_dataset(rid, split="validation", streaming=True)
        for i, ex in enumerate(ds):
            if i >= limit_per_source:
                break
            stem = f"{name}_{i:06d}"
            p = OUT / "images" / f"{stem}.png"
            ex["image"].convert("RGB").save(p)
            records.append({
                "dataset": name, "id": stem, "task": "vqa",
                "images": [str(p)],
                "question": ex["question"].strip(),
                "answer": str(ex["answer"]).strip(),
            })
            if (i + 1) % 100 == 0:
                print(f"  {name}: {i + 1} saved", flush=True)
        print(f"  {name}: finished, {len([r for r in records if r['dataset'] == name])} records", flush=True)

    out = OUT / "records.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWROTE {out}  ({len(records)} records)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-source", type=int, default=2500)
    main(ap.parse_args().limit_per_source)
