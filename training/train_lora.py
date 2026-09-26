"""ML-08: QLoRA fine-tune of EarthDial 4B on the SatQuery remote-sensing mix.

    python training/train_lora.py --smoke          # 10 steps, validates the pipeline
    python training/train_lora.py                  # the real run

Reuses EarthDial's own preprocessing -- preprocess_phi3 for the token surgery
(<image> -> <img> + N x <IMG_CONTEXT> + </img>, prompt masked out of the labels)
and build_transform for the pixels -- because every one of those details is
silently wrong if reimplemented, and a silent error here looks like a loss that
simply does not fall.

Deviation from the handover: bf16, not fp16. The handover specifies fp16 because
a T4 has no bf16. This trains on an H100 (capability 9.0), where bf16 is native,
faster and more stable for LoRA.
"""

from __future__ import annotations

import argparse, json, sys, types
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

EARTHDIAL = Path.home() / "EarthDial" / "src"
sys.path.insert(0, str(EARTHDIAL))

# their dataset module imports decord (video only) at module level
if "decord" not in sys.modules:
    stub = types.ModuleType("decord")
    stub.VideoReader = object
    stub.cpu = lambda *a, **k: None
    sys.modules["decord"] = stub

from transformers import (AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)
from earthdial.model.internvl_chat import InternVLChatModel
from earthdial.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from earthdial.train.constants import IMG_CONTEXT_TOKEN
from earthdial.train.dataset import build_transform, preprocess_phi3

RID = "akshaydudhane/EarthDial_4B_RGB"
TEMPLATE = "phi3-chat"


class AdapterTrainer(Trainer):
    """Checkpoint the LoRA adapter, not the frozen 4-bit base.

    wrap_llm_lora attaches adapters to model.language_model, so transformers'
    default _save looks for model.peft_config on the top-level InternVLChatModel
    and fails -- and if it succeeded it would try to serialise 4B quantized
    parameters every 500 steps.
    """

    def _save(self, output_dir=None, state_dict=None):
        out = Path(output_dir or self.args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.language_model.save_pretrained(out)
        tok = getattr(self, "_satquery_tokenizer", None)
        if tok is not None:
            tok.save_pretrained(out)
        print(f"  checkpoint -> {out}", flush=True)
        repo = os.environ.get("SATQUERY_HF_ADAPTER_REPO")
        if repo:
            try:
                from huggingface_hub import upload_folder
                upload_folder(repo_id=repo, folder_path=str(out), repo_type="model",
                              commit_message=f"checkpoint {out.name}")
                print(f"  backed up to HF: {repo}", flush=True)
            except Exception as e:
                print(f"  HF upload failed (kept local): {e}", flush=True)


def load_model(image_size, four_bit=True):
    InternVLChatConfig.has_no_defaults_at_init = True
    if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
        InternVLChatModel.all_tied_weights_keys = {}

    cfg = InternVLChatConfig.from_pretrained(RID)
    rs = getattr(cfg.llm_config, "rope_scaling", None)
    if rs:
        for k in ("type", "rope_type"):
            if rs.get(k) == "su":
                rs[k] = "longrope"          # transformers renamed it
        rs.setdefault("rope_type", rs.get("type", "longrope"))
        rs.setdefault("type", rs.get("rope_type"))
        cfg.llm_config.rope_scaling = rs
        if getattr(cfg.llm_config, "original_max_position_embeddings", None) is None:
            cfg.llm_config.original_max_position_embeddings = 4096

    kw = {}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = InternVLChatModel.from_pretrained(
        RID, config=cfg, torch_dtype=torch.bfloat16, device_map={"": 0}, **kw)
    return model


class SatQueryDataset(Dataset):
    def __init__(self, path, tokenizer, num_image_token, image_size, max_len):
        self.rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        self.tok = tokenizer
        self.n_img_tok = num_image_token
        self.transform = build_transform(is_train=False, input_size=image_size,
                                         normalize_type="imagenet")
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rec = self.rows[i]
        paths = rec["image"] if isinstance(rec["image"], list) else [rec["image"]]
        pixel_values = torch.stack(
            [self.transform(Image.open(p).convert("RGB")) for p in paths])

        ret = preprocess_phi3(
            TEMPLATE, [deepcopy(rec["conversations"])], self.tok,
            [self.n_img_tok] * len(paths), num_image=len(paths))

        return {
            "input_ids": ret["input_ids"][0][: self.max_len],
            "labels": ret["labels"][0][: self.max_len],
            "attention_mask": ret["attention_mask"][0][: self.max_len],
            "pixel_values": pixel_values.to(torch.bfloat16),
            "image_flags": torch.ones(len(paths), 1, dtype=torch.long),
        }


def collate(batch, pad_id):
    longest = max(b["input_ids"].shape[0] for b in batch)

    def pad(x, fill):
        need = longest - x.shape[0]
        return x if need == 0 else torch.cat([x, torch.full((need,), fill, dtype=x.dtype)])

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_id) for b in batch]),
        "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
        "image_flags": torch.cat([b["image_flags"] for b in batch]),
    }


def main(a):
    print("=== tokenizer ===", flush=True)
    tok = AutoTokenizer.from_pretrained(RID, trust_remote_code=True, use_fast=False)
    tok.add_tokens([IMG_CONTEXT_TOKEN], special_tokens=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.model_max_length = a.max_len
    img_ctx_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print(f"  IMG_CONTEXT id = {img_ctx_id}, pad = {tok.pad_token!r}", flush=True)

    print("\n=== model ===", flush=True)
    model = load_model(a.image_size, four_bit=not a.no_4bit)
    model.img_context_token_id = img_ctx_id
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    print(f"  loaded, {torch.cuda.memory_allocated()/1024**3:.2f} GiB, image_size={image_size}", flush=True)

    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.mlp1.parameters():
        p.requires_grad = False

    model.wrap_llm_lora(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout)
    model.config.use_cache = False
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.3f}%)", flush=True)

    print("\n=== data ===", flush=True)
    ds = SatQueryDataset("training/train.jsonl", tok, model.num_image_token,
                         image_size, a.max_len)
    eval_ds = None
    if Path("training/heldout.jsonl").exists() and not a.smoke:
        eval_ds = SatQueryDataset("training/heldout.jsonl", tok, model.num_image_token,
                                  image_size, a.max_len)
        eval_ds.rows = eval_ds.rows[:300]
        print(f"  eval set: {len(eval_ds)} held-out samples", flush=True)
    if a.smoke:
        ds.rows = ds.rows[:64]
    print(f"  {len(ds)} samples", flush=True)
    import random as _rnd
    for i in _rnd.Random(0).sample(range(len(ds)), min(6, len(ds))):
        s_ = ds[i]
        print(f"  sample {i:5d} task={ds.rows[i]['task']:11s} "
              f"images={tuple(s_['pixel_values'].shape)} "
              f"len={tuple(s_['input_ids'].shape)} "
              f"supervised={int((s_['labels'] != -100).sum())}", flush=True)

    desired = dict(
        output_dir=a.out,
        num_train_epochs=1,
        max_steps=10 if a.smoke else -1,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=max(1, a.effective_batch // a.batch),
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=a.log_every,
        save_steps=500,
        eval_strategy="steps" if not a.smoke else "no",
        eval_steps=200,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        report_to=[],
        # InternVLChatModel does not declare supports_gradient_checkpointing,
        # and we do not need it: 2.14 GiB of weights in a 19.6 GiB slice.
        # Off is also faster -- checkpointing trades compute for memory.
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_num_workers=a.workers,
        seed=42,
    )
    import inspect
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(k for k in desired if k not in supported)
    if dropped:
        print(f"  TrainingArguments: dropping unsupported {dropped}", flush=True)
    args = TrainingArguments(**{k: v for k, v in desired.items() if k in supported})
    # transformers 5.x refuses to fine-tune a quantized model unless it can see
    # PEFT adapters. wrap_llm_lora attaches them to model.language_model, one
    # level below where the guard looks, so it misses 25M genuinely trainable
    # parameters. This flag is how transformers marks "adapters are present".
    model._hf_peft_config_loaded = True

    trainer = AdapterTrainer(model=model, args=args, train_dataset=ds,
                             eval_dataset=eval_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer._satquery_tokenizer = tok

    print("\n=== training ===", flush=True)
    trainer.train()

    out = Path(a.out) / "lora-v1"
    out.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"\nSAVED adapter -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/adapters")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args())
