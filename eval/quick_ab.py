"""Base EarthDial vs EarthDial+LoRA on real held-out samples.

    python eval/quick_ab.py --n 12                # with the adapter
    python eval/quick_ab.py --n 12 --no-adapter   # base model

Diagnostic, not a benchmark: the held-out set is drawn from our own training
mix, so it shows behaviour, not capability. Capability needs the official test
splits.
"""

from __future__ import annotations

import argparse, json, os, random
from pathlib import Path


def main(a):
    if a.no_adapter:
        os.environ["SATQUERY_ADAPTER"] = "/nonexistent"  # _adapter_dir() -> None

    import yaml
    from satquery.common.paths import REPO_ROOT
    from satquery.contracts import (ImageInfo, JobContext, SensorProfile,
                                    ToolCall, ToolCard)
    from satquery.models.rs_vlm import RSVLM

    rows = [json.loads(l) for l in Path("training/heldout.jsonl").read_text().splitlines() if l.strip()]
    by_ds: dict[str, list] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)

    rng = random.Random(42)
    picked = []
    per = max(1, a.n // len(by_ds))
    for ds, items in sorted(by_ds.items()):
        picked.extend(rng.sample(items, min(per, len(items))))

    card = ToolCard.model_validate(
        yaml.safe_load((REPO_ROOT / "registry" / "rs_vlm.yaml").read_text()))
    tool = RSVLM(card)

    print(f"=== {'BASE (no adapter)' if a.no_adapter else 'WITH LoRA'} ===", flush=True)
    for r in picked:
        paths = [Path(p) for p in r["image"]]
        if not all(p.exists() for p in paths):
            continue
        images, roles = [], ["t1", "t2"] if len(paths) == 2 else ["image"]
        for path, role in zip(paths, roles):
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
            images.append(ImageInfo(image_id=path.stem, role="T1" if role == "t1" else
                                    ("T2" if role == "t2" else "single"),
                                    file=path.name, sha1="0" * 40,
                                    prepared_path=path, rgb8_path=path,
                                    pixel_m=None, width=w, height=h))
        ctx = JobContext(job_id="ab", workdir=Path("/tmp/ab"),
                         profile=SensorProfile(input_mode="benchmark", images=images),
                         mode="real")

        human = r["conversations"][0]["value"]
        gold = r["conversations"][1]["value"]
        mode = {"caption": "caption", "vqa": "vqa", "change_vqa": "describe_change"}[r["task"]]
        question = human.replace("<image>", "").replace("Image 1:", "").replace("Image 2:", "").strip()

        inputs = {"t1": "T1", "t2": "T2"} if len(images) == 2 else {"image": "single"}
        call = ToolCall(step_id="s1", tool="rs_vlm", inputs=inputs,
                        params={"mode": mode, "question": question, "max_new_tokens": 64})
        try:
            res = tool.run_real(call, ctx)
            pred, score = res.text, res.score
        except Exception as exc:
            pred, score = f"<failed: {type(exc).__name__}: {str(exc)[:80]}>", None

        print(f"\n[{r['dataset']}/{r['task']}]", flush=True)
        print(f"  Q    : {question[:150]}", flush=True)
        print(f"  GOLD : {gold[:150]}", flush=True)
        print(f"  PRED : {pred[:150]}   (score {score})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--no-adapter", action="store_true")
    main(ap.parse_args())
