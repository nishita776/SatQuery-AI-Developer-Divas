"""Measure rs_vlm accuracy on the VRSBench VQA test set: base vs +LoRA.

    python eval/run_vqa.py --n 300

Prints two accuracy numbers on the SAME questions, so the difference is the
adapter's effect and neither number can be cherry-picked. Also checks that no
test image appears in the training data -- if any do, the numbers are void.
"""

from __future__ import annotations

import argparse, json, os, random, re, string, zipfile
from pathlib import Path

RID = "xiang709/VRSBench"


def normalise(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def training_basenames() -> set[str]:
    names = set()
    for f in Path("training/data/raw").glob("*/records.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            for img in json.loads(line).get("images", []):
                names.add(Path(img).name)
    return names


def run_pass(items, images_dir, use_adapter):
    """Return (correct, total, examples). Loads the model once."""
    import importlib
    import satquery.models.rs_vlm as mod

    if not use_adapter:
        os.environ["SATQUERY_ADAPTER"] = "/nonexistent"  # hide the adapter
    else:
        os.environ.pop("SATQUERY_ADAPTER", None)
    importlib.reload(mod)  # reset the module-level model cache

    import yaml
    from satquery.common.paths import REPO_ROOT
    from satquery.contracts import (ImageInfo, JobContext, SensorProfile,
                                    ToolCall, ToolCard)
    from PIL import Image

    card = ToolCard.model_validate(
        yaml.safe_load((REPO_ROOT / "registry" / "rs_vlm.yaml").read_text()))
    tool = mod.RSVLM(card)

    correct, total, examples = 0, 0, []
    label = "WITH LoRA" if use_adapter else "BASE     "
    for i, it in enumerate(items):
        path = images_dir / it["image_id"]
        if not path.exists():
            continue
        with Image.open(path) as im:
            w, h = im.size
        info = ImageInfo(image_id=path.stem, role="single", file=path.name,
                         sha1="0" * 40, prepared_path=path, rgb8_path=path,
                         pixel_m=None, width=w, height=h)
        ctx = JobContext(job_id="eval", workdir=Path("/tmp/eval"),
                         profile=SensorProfile(input_mode="benchmark", images=[info]),
                         mode="real")
        call = ToolCall(step_id="s1", tool="rs_vlm", inputs={"image": "single"},
                        params={"mode": "vqa", "question": it["question"],
                                "max_new_tokens": 32})
        try:
            pred = tool.run_real(call, ctx).text
        except Exception as exc:
            pred = f"<error: {type(exc).__name__}>"
        hit = normalise(pred) == normalise(it["ground_truth"])
        correct += hit
        total += 1
        if len(examples) < 8:
            examples.append((it["question"], it["ground_truth"], pred, hit))
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i + 1}/{len(items)}  running acc "
                  f"{100 * correct / max(total, 1):.1f}%", flush=True)
    return correct, total, examples


def main(a):
    from huggingface_hub import hf_hub_download

    print("=== loading test questions ===", flush=True)
    ann = hf_hub_download(RID, "VRSBench_EVAL_vqa.json", repo_type="dataset")
    items = json.loads(Path(ann).read_text())
    random.Random(42).shuffle(items)
    items = items[: a.n]
    print(f"  {len(items)} questions", flush=True)

    print("\n=== leakage check ===", flush=True)
    train = training_basenames()
    val_used = {it["image_id"] for it in items}
    leaked = val_used & train
    print(f"  {len(train)} training images, {len(val_used)} test images used, "
          f"OVERLAP: {len(leaked)}", flush=True)
    if leaked:
        print(f"  WARNING leaked: {sorted(leaked)[:10]}", flush=True)

    print("\n=== extracting the test images ===", flush=True)
    images_dir = Path("eval/vrsbench_val_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in images_dir.glob("*.png")}
    need = val_used - have
    if need:
        zip_path = hf_hub_download(RID, "Images_val.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as zf:
            members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
            for name in need:
                if name in members:
                    with zf.open(members[name]) as src, (images_dir / name).open("wb") as fh:
                        fh.write(src.read())
    print(f"  {len(list(images_dir.glob('*.png')))} images ready", flush=True)

    print("\n=== BASE MODEL ===", flush=True)
    b_correct, b_total, b_ex = run_pass(items, images_dir, use_adapter=False)

    print("\n=== WITH LoRA ===", flush=True)
    a_correct, a_total, a_ex = run_pass(items, images_dir, use_adapter=True)

    b_acc = 100 * b_correct / max(b_total, 1)
    a_acc = 100 * a_correct / max(a_total, 1)

    print("\n" + "=" * 60, flush=True)
    print(f"  VRSBench VQA, {b_total} questions", flush=True)
    print(f"  BASE      : {b_acc:.1f}%  ({b_correct}/{b_total})", flush=True)
    print(f"  WITH LoRA : {a_acc:.1f}%  ({a_correct}/{a_total})", flush=True)
    print(f"  DELTA     : {a_acc - b_acc:+.1f} points", flush=True)
    print(f"  leakage   : {len(leaked)} test images seen in training", flush=True)
    print("=" * 60, flush=True)

    Path("eval").mkdir(exist_ok=True)
    Path("eval/results_vqa.json").write_text(json.dumps({
        "benchmark": "VRSBench-VQA", "n": b_total,
        "base_acc": round(b_acc, 2), "lora_acc": round(a_acc, 2),
        "delta": round(a_acc - b_acc, 2), "leakage": len(leaked),
        "base_examples": b_ex, "lora_examples": a_ex,
    }, indent=2))
    print("\nwrote eval/results_vqa.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    main(ap.parse_args())
