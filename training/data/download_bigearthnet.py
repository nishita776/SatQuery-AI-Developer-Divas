"""ML-07: BigEarthNet -> training/data/raw/bigearthnet/ (images + records.jsonl).

danielz01/BigEarthNet-S2-v1.0, config s2-rgb: Sentinel-2 patches already
composited to 8-bit RGB (120x120, 10 m/px) with CORINE land-cover `labels`.

Why text is generated from labels: BigEarthNet ships labels, not captions.
BigEarthNet.txt (BIFOLD, CDLA-Permissive-1.0) has 9.6M real annotations but is
text-only and keys on a patch_id this mirror does not expose, so it cannot be
joined here. Labels give the land-cover vocabulary, which is why BigEarthNet is
in the mix at all. Seed 42 (master F3.2).
"""

import argparse, json, random
from pathlib import Path
from datasets import load_dataset

RID = "danielz01/BigEarthNet-S2-v1.0"
CONFIG = "s2-rgb"
OUT = Path("training/data/raw/bigearthnet")
SEED = 42

CAPTION_TEMPLATES = [
    "A satellite image showing {}.",
    "This scene mainly contains {}.",
    "An aerial view dominated by {}.",
    "The land cover here is {}.",
    "Sentinel-2 imagery of {}.",
    "The area is largely {}.",
    "Overhead imagery of {}.",
    "This region is covered by {}.",
    "A remote-sensing image of {}.",
    "The dominant land cover is {}.",
]
QUESTIONS = [
    "What land cover types are present in this image?",
    "Describe the land cover visible here.",
    "Which land cover classes can you identify?",
    "What is the dominant land cover in this scene?",
    "What kind of terrain does this image show?",
    "List the land cover types in this satellite image.",
]


def phrase(labels):
    items = [str(x).strip().lower() for x in labels if str(x).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def main(n_patches):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ds = load_dataset(RID, CONFIG, split="train", streaming=True)

    records, skipped = [], 0
    print(f"=== BigEarthNet {CONFIG}, target {n_patches} patches ===", flush=True)
    for i, ex in enumerate(ds):
        if len(records) // 2 >= n_patches:
            break
        text = phrase(ex.get("labels") or [])
        if not text:
            skipped += 1
            continue
        stem = f"ben_{len(records) // 2:06d}"
        p = OUT / "images" / f"{stem}.png"
        ex["img"].convert("RGB").save(p)

        records.append({
            "dataset": "bigearthnet", "id": f"{stem}_cap", "task": "caption",
            "images": [str(p)], "question": "",
            "answer": rng.choice(CAPTION_TEMPLATES).format(text),
        })
        records.append({
            "dataset": "bigearthnet", "id": f"{stem}_qa", "task": "vqa",
            "images": [str(p)], "question": rng.choice(QUESTIONS),
            "answer": text.capitalize() + ".",
        })
        if len(records) % 200 == 0:
            print(f"  {len(records) // 2} patches, {len(records)} records "
                  f"(skipped {skipped})", flush=True)

    out = OUT / "records.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWROTE {out}  ({len(records)} records from "
          f"{len(records) // 2} patches, {skipped} unlabelled)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patches", type=int, default=10000)
    main(ap.parse_args().n_patches)
