"""ML-07: VRSBench -> training/data/raw/vrsbench/ (images + records.jsonl).

VRSBench_train.json is LLaVA format: one conversation per item, ~142k items over
~29k images, with a task tag in the human turn ('[caption] ...', '[vqa] ...').

Grounding/referring items are dropped on purpose: master F4.3 says the
controller never asks rs_vlm for boxes -- that is the detector's job, and the
detector is not fine-tuned. Seed 42 (master F3.2).
"""

import argparse, json, random, re, zipfile
from collections import Counter
from pathlib import Path
from huggingface_hub import hf_hub_download

RID = "xiang709/VRSBench"
OUT = Path("training/data/raw/vrsbench")
SEED = 42
TAG = re.compile(r"\[([a-zA-Z_ ]+)\]")
KEEP = {"caption": "caption", "vqa": "vqa"}


def parse(item):
    """-> (task, question, answer) or None if we do not want this item."""
    convs = item.get("conversations") or []
    human = next((c for c in convs if str(c.get("from")).lower() in ("human", "user")), None)
    gpt = next((c for c in convs if str(c.get("from")).lower() in ("gpt", "assistant")), None)
    if not human or not gpt:
        return None
    raw = str(human.get("value", ""))
    m = TAG.search(raw)
    tag = (m.group(1).strip().lower() if m else "")
    task = KEEP.get(tag)
    if task is None:
        return None
    q = TAG.sub("", raw).replace("<image>", "").strip()
    q = re.sub(r"^\s*The question\s+(.*?)\s+can be answered using the image\.?\s*"
               r"A short answer is\.?\s*$", r"\1", q, flags=re.I | re.S).strip()
    a = str(gpt.get("value", "")).strip()
    return (task, q, a) if a else None


def main(captions, vqa):
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    print("=== annotations ===", flush=True)
    ann = hf_hub_download(RID, "VRSBench_train.json", repo_type="dataset")
    items = json.loads(Path(ann).read_text())
    print(f"  {len(items)} items", flush=True)

    tags = Counter()
    for it in items:
        h = next((c for c in it.get("conversations") or []
                  if str(c.get("from")).lower() in ("human", "user")), None)
        m = TAG.search(str(h.get("value", ""))) if h else None
        tags[(m.group(1).strip().lower() if m else "(none)")] += 1
    print(f"  task tags present: {dict(tags)}", flush=True)

    random.Random(SEED).shuffle(items)
    quota = {"caption": captions, "vqa": vqa}
    taken = {"caption": 0, "vqa": 0}
    chosen = []
    for it in items:
        if sum(taken.values()) >= sum(quota.values()):
            break
        parsed = parse(it)
        if not parsed:
            continue
        task = parsed[0]
        if taken[task] >= quota[task]:
            continue
        taken[task] += 1
        chosen.append((it, parsed))
    print(f"  kept {len(chosen)} items: {taken}", flush=True)

    wanted = {Path(str(it.get("image", ""))).name for it, _ in chosen}
    print(f"  {len(wanted)} unique images needed", flush=True)

    have = {p.name for p in (OUT / "images").glob("*")}
    todo = sorted(wanted - have)
    print(f"  {len(have)} already extracted, {len(todo)} to go", flush=True)

    if todo:
        print("\n=== extracting from the cached zip ===", flush=True)
        zip_path = hf_hub_download(RID, "Images_train.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
            for i, name in enumerate(todo, 1):
                member = members.get(name)
                if member is None:
                    continue
                with zf.open(member) as src, (OUT / "images" / name).open("wb") as fh:
                    fh.write(src.read())
                if i % 250 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    print("\n=== records ===", flush=True)
    records, missing = [], 0
    for it, (task, q, a) in chosen:
        name = Path(str(it.get("image", ""))).name
        dest = OUT / "images" / name
        if not dest.exists():
            missing += 1
            continue
        records.append({
            "dataset": "vrsbench", "id": f"{Path(name).stem}_{task}_{len(records)}",
            "task": task, "images": [str(dest)], "question": q, "answer": a,
        })

    out = OUT / "records.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    by_task = Counter(r["task"] for r in records)
    print(f"\nWROTE {out}", flush=True)
    print(f"  {len(records)} records: {dict(by_task)}", flush=True)
    print(f"  {missing} had no image on disk", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", type=int, default=5000)
    ap.add_argument("--vqa", type=int, default=5000)
    a = ap.parse_args()
    main(a.captions, a.vqa)
