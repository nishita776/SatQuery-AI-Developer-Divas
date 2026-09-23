"""ML-07: CDVQA -> training/data/raw/cdvqa/ (image pairs + records.jsonl).

WebDataset shards: each record is two PNGs plus a `json` field whose
`conversations` are already in chat format, with 'Image 1: <image>' /
'Image 2: <image>' markers. Two-image change VQA, which is what
rs_vlm's describe_change and CHANGE_VQA modes need.
"""

import argparse, json, re, sys
from pathlib import Path
from datasets import load_dataset

RID = "ljx620/CDVQA"
OUT = Path("training/data/raw/cdvqa")

def split_conversation(convs):
    """-> (question, answer), image markers stripped."""
    q = a = None
    for turn in convs:
        who = str(turn.get("from", "")).lower()
        val = str(turn.get("value", ""))
        val = re.sub(r"Image\s*\d+\s*:\s*<image>\s*", "", val).strip()
        if who in ("user", "human") and q is None:
            q = val
        elif who in ("assistant", "gpt") and a is None:
            a = val
    return q, a

def main(limit):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(RID, split="train", streaming=True)
    records, skipped = [], 0
    print(f"=== CDVQA ({RID}), target {limit} ===", flush=True)
    for i, ex in enumerate(ds):
        if len(records) >= limit:
            break
        convs = (ex.get("json") or {}).get("conversations") or []
        q, a = split_conversation(convs)
        if not q or not a:
            skipped += 1
            continue
        stem = f"cdvqa_{len(records):06d}"
        pa = OUT / "images" / f"{stem}_t1.png"
        pb = OUT / "images" / f"{stem}_t2.png"
        pa.write_bytes(ex["0.img"])
        pb.write_bytes(ex["1.img"])
        records.append({
            "dataset": "cdvqa", "id": stem, "task": "change_vqa",
            "images": [str(pa), str(pb)], "question": q, "answer": a,
        })
        if len(records) % 100 == 0:
            print(f"  {len(records)} saved  (skipped {skipped})", flush=True)

    out = OUT / "records.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWROTE {out}  ({len(records)} records, {skipped} skipped)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000)
    main(ap.parse_args().limit)
