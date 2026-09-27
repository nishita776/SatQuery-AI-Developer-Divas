"""GeoMMBench MCQ eval: base EarthDial vs +LoRA. Mirrors run_vqa.py."""
import argparse, importlib, json, os, re
from pathlib import Path
from PIL import Image

def run_pass(items, images_dir, use_adapter):
    import satquery.models.rs_vlm as mod
    if not use_adapter:
        os.environ["SATQUERY_ADAPTER"] = "/nonexistent"
    else:
        os.environ.pop("SATQUERY_ADAPTER", None)
    importlib.reload(mod)
    import yaml
    from satquery.common.paths import REPO_ROOT
    from satquery.contracts import (ImageInfo, JobContext, SensorProfile,
                                    ToolCall, ToolCard)
    card = ToolCard.model_validate(
        yaml.safe_load((REPO_ROOT / "registry" / "rs_vlm.yaml").read_text()))
    tool = mod.RSVLM(card)
    correct, total = 0, 0
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
        ctx = JobContext(job_id="eval", workdir=Path("/tmp/geomm"),
                         profile=SensorProfile(input_mode="benchmark", images=[info]),
                         mode="real")
        q = it["question"] + "\n" + it["choices"] + "\nAnswer with the letter only."
        call = ToolCall(step_id="s1", tool="rs_vlm", inputs={"image": "single"},
                        params={"mode": "vqa", "question": q, "max_new_tokens": 8})
        try:
            pred = tool.run_real(call, ctx).text
        except Exception:
            pred = ""
        m = re.search(r"[ABCD]", pred.upper())
        got = m.group(0) if m else "?"
        total += 1
        if got == it["answer"].upper():
            correct += 1
        if total % 25 == 0:
            print(f"  [{label}] {total}/{len(items)}  running acc "
                  f"{100*correct/total:.1f}%", flush=True)
    return correct, total

def main(args):
    from datasets import load_dataset
    print("=== loading GeoMMBench ===", flush=True)
    ds = load_dataset("AR-X/GeoMMBench", split="test")
    images_dir = Path("eval/geommbench_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    items = []
    letters = ["A", "B", "C", "D"]
    for i, row in enumerate(ds):
        if len(items) >= args.n:
            break
        img = row.get("image")
        if img is None:
            continue
        name = f"gmb_{i}.png"
        p = images_dir / name
        if not p.exists():
            img.convert("RGB").save(p)
        opts = [row.get(k) for k in ("A", "B", "C", "D")]
        if any(o is None for o in opts):
            # some GeoMMBench variants pack options in a list field
            opts = row.get("options") or row.get("choices")
            if not opts:
                continue
        choices = "  ".join(f"{L}. {o}" for L, o in zip(letters, opts))
        ans = row.get("answer") or row.get("label")
        if ans is None:
            continue
        ans = str(ans).strip().upper()[:1]
        items.append({"image_id": name, "question": str(row.get("question", "")),
                      "choices": choices, "answer": ans})
    print(f"  {len(items)} MCQ items", flush=True)
    print("\n=== BASE MODEL ===", flush=True)
    bc, bt = run_pass(items, images_dir, use_adapter=False)
    print("\n=== WITH LoRA ===", flush=True)
    ac, at = run_pass(items, images_dir, use_adapter=True)
    base = 100 * bc / bt if bt else 0
    lora = 100 * ac / at if at else 0
    print("\n" + "=" * 60)
    print(f"  GeoMMBench MCQ, {bt} questions")
    print(f"  BASE      : {base:.1f}%  ({bc}/{bt})")
    print(f"  WITH LoRA : {lora:.1f}%  ({ac}/{at})")
    print(f"  DELTA     : {lora-base:+.1f} points")
    print("=" * 60)
    out = {"n": bt, "base": base, "lora": lora, "delta": lora - base}
    Path("eval/results_geommbench.json").write_text(json.dumps(out, indent=2))
    print("wrote eval/results_geommbench.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    main(ap.parse_args())
