"""ML-07: BigEarthNet Sentinel-1 -> training/data/raw/bigearthnet_s1/.

Radar half of the PS-mandated dataset, so the adapter is trained on SAR as well
as optical -- "our adapter saw radar" is a claim we can defend.

Two sources joined on the S1 patch name:
  torchgeo/bigearthnet  V2/BigEarthNet-S1  per-band float GeoTIFFs (VH, VV)
  BIFOLD .../BigEarthNet.txt               9.6M real annotations, with s1_name

Images are composited to a false-colour RGB the way SAR normally is read:
  R = VV, G = VH, B = VV - VH
each stretched 2-98% to 8-bit, matching what ingest hands the tools at
inference (master F3.2). Seed 42.
"""

from __future__ import annotations

import argparse, io, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download

IMG_RID = "torchgeo/bigearthnet"
TXT_RID = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
OUT = Path("training/data/raw/bigearthnet_s1")


def stretch(a):
    """2-98% percentile stretch to uint8 (master F3.2 / ML-07 step 4)."""
    a = np.asarray(a, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def composite(vv, vh):
    return np.dstack([stretch(vv), stretch(vh), stretch(vv - vh)])


def patch_name(key: str) -> tuple[str, str]:
    """'.../S1A_..._61_39/S1A_..._61_39_VH' -> ('S1A_..._61_39', 'VH')"""
    parts = key.split("/")
    band = parts[-1].rsplit("_", 1)[-1].upper()
    return parts[-2] if len(parts) >= 2 else parts[-1], band


def main(n_patches, qa_per_patch):
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    print(f"=== streaming {IMG_RID} for {n_patches} S1 patches ===", flush=True)
    ds = load_dataset(IMG_RID, split="train", streaming=True)
    pending: dict[str, dict] = defaultdict(dict)
    saved: dict[str, Path] = {}

    for rec in ds:
        key = str(rec.get("__key__", ""))
        if "BigEarthNet-S1" not in key:
            continue
        name, band = patch_name(key)
        if band not in ("VV", "VH") or name in saved:
            continue
        pending[name][band] = np.array(rec["tif"], dtype=np.float32)
        if len(pending[name]) < 2:
            continue
        rgb = composite(pending[name]["VV"], pending[name]["VH"])
        p = OUT / "images" / f"{name}.png"
        Image.fromarray(rgb).save(p)
        saved[name] = p
        pending.pop(name, None)
        if len(saved) % 100 == 0:
            print(f"  {len(saved)}/{n_patches} patches", flush=True)
        if len(saved) >= n_patches:
            break

    print(f"  composited {len(saved)} patches", flush=True)
    if not saved:
        raise SystemExit("no S1 patches found -- check the stream key layout")

    print(f"\n=== joining to {TXT_RID} on s1_name ===", flush=True)
    import pyarrow.parquet as pq

    pf = hf_hub_download(TXT_RID, "BigEarthNet.txt.parquet", repo_type="dataset")
    tbl = pq.read_table(pf, columns=["s1_name", "input", "output", "split", "category"])
    df = tbl.to_pandas()
    print(f"  {len(df)} annotations total", flush=True)
    df = df[(df["split"] == "train") & (df["s1_name"].isin(saved.keys()))]
    print(f"  {len(df)} annotations match our patches (train split only)", flush=True)

    records = []
    for name, grp in df.groupby("s1_name"):
        img = str(saved[name])
        for _, row in grp.head(qa_per_patch).iterrows():
            q, a = str(row["input"]).strip(), str(row["output"]).strip()
            if q and a:
                records.append({
                    "dataset": "bigearthnet_s1", "id": f"{name}_{len(records):06d}",
                    "task": "vqa", "images": [img], "question": q, "answer": a,
                })

    out = OUT / "records.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWROTE {out}  ({len(records)} records over "
          f"{df['s1_name'].nunique()} patches)", flush=True)
    if not records:
        print("  WARNING: zero matches -- the s1_name formats may differ", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patches", type=int, default=5000)
    ap.add_argument("--qa-per-patch", type=int, default=2)
    a = ap.parse_args()
    main(a.n_patches, a.qa_per_patch)
