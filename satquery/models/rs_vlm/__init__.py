"""rs_vlm -- remote-sensing VLM. EarthDial 4B RGB in 4-bit (see DECISION.md).

Four modes (master F4.3): vqa, caption, describe_change, describe_scene. It is
never asked for counts or boxes -- that is the detector's job, and the card does
not list COUNT or GROUNDING.

No double tiling. EarthDial does its own dynamic tiling at 448 px (up to
max_dynamic_patch crops plus a thumbnail), so it gets a region crop and handles
the rest itself. The Scale Bridge serves the detector and change_map.

This mirrors EarthDial's own chat() rather than calling it, because chat()
returns decoded text only and master F3.5 defines score as the mean probability
of the generated tokens -- which needs generate(output_scores=True).

Requires the EarthDial source, patched by training/patch_earthdial.py. Point
SATQUERY_EARTHDIAL_SRC at it if it is not at ~/EarthDial/src.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from satquery.common.toolbase import BaseTool
from satquery.contracts import (ErrorCode, JobContext, ToolCall, ToolError,
                                ToolResult)
from satquery.models.rs_vlm import prompts

__all__ = ["RSVLM", "prompts"]

log = logging.getLogger("satquery.models")

BACKBONE = "akshaydudhane/EarthDial_4B_RGB"
_CACHE: dict = {}


def _earthdial_src() -> Path:
    return Path(os.environ.get("SATQUERY_EARTHDIAL_SRC", Path.home() / "EarthDial" / "src"))


def _adapter_dir() -> Path | None:
    p = Path(os.environ.get("SATQUERY_ADAPTER", "training/adapters/lora-v1"))
    return p if (p / "adapter_config.json").exists() else None


def _load():
    """Lazy and cached. Every model import lives in here (master F3.4)."""
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["tokenizer"], _CACHE["model_id"]

    src = _earthdial_src()
    if not src.exists():
        raise ToolError(ErrorCode.MODEL_LOAD_FAILED,
                        f"EarthDial source not found at {src}; set SATQUERY_EARTHDIAL_SRC")
    import sys
    import types
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if "decord" not in sys.modules:  # video-only dep of their dataset module
        stub = types.ModuleType("decord")
        stub.VideoReader = object
        stub.cpu = lambda *a, **k: None
        sys.modules["decord"] = stub

    try:
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from earthdial.model.internvl_chat import InternVLChatModel
        from earthdial.model.internvl_chat.configuration_internvl_chat import \
            InternVLChatConfig
    except ImportError as exc:
        raise ToolError(ErrorCode.MODEL_LOAD_FAILED, f"cannot import EarthDial: {exc}") from exc

    try:
        InternVLChatConfig.has_no_defaults_at_init = True
        if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
            InternVLChatModel.all_tied_weights_keys = {}

        cfg = InternVLChatConfig.from_pretrained(BACKBONE)
        rs = getattr(cfg.llm_config, "rope_scaling", None)
        if rs:  # transformers renamed this rope type
            for k in ("type", "rope_type"):
                if rs.get(k) == "su":
                    rs[k] = "longrope"
            rs.setdefault("rope_type", rs.get("type", "longrope"))
            rs.setdefault("type", rs.get("rope_type"))
            cfg.llm_config.rope_scaling = rs
            if getattr(cfg.llm_config, "original_max_position_embeddings", None) is None:
                cfg.llm_config.original_max_position_embeddings = 4096

        # bf16 where the hardware has it (H100), fp16 on Turing (T4)
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
        dtype = torch.bfloat16 if native_bf16 else torch.float16

        model = InternVLChatModel.from_pretrained(
            BACKBONE, config=cfg, torch_dtype=dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
            device_map={"": 0} if torch.cuda.is_available() else None,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE, trust_remote_code=True, use_fast=False)
    except Exception as exc:
        raise ToolError(ErrorCode.MODEL_LOAD_FAILED, f"could not load {BACKBONE}: {exc}") from exc

    model_id = "earthdial-4b-rgb-4bit"
    adapter = _adapter_dir()
    if adapter is not None:
        try:
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(model.language_model, str(adapter))
            model_id = f"{model_id}+{adapter.name}"
            log.info("rs_vlm loaded adapter %s", adapter)
        except Exception as exc:
            log.warning("adapter at %s failed to load: %s", adapter, exc)

    # EarthDial forwards return_dict into generate(); v5's _prefill supplies it
    lm = model.language_model
    if not getattr(lm, "_return_dict_shim", False):
        original = lm.generate

        def _generate(*args, **kwargs):
            kwargs.pop("return_dict", None)
            return original(*args, **kwargs)

        lm.generate = _generate
        lm._return_dict_shim = True

    _CACHE.update(model=model, tokenizer=tokenizer, model_id=model_id)
    return model, tokenizer, model_id


def _crop_window(image, region):
    """Region bounding box plus a 10% margin (master ML-06 step 2), in ref pixels."""
    import numpy as np

    from satquery.common.geo import read_mask

    mask = read_mask(region)
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ToolError(ErrorCode.BAD_INPUT, "region selects no pixels")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    my, mx = int(0.1 * (y1 - y0)), int(0.1 * (x1 - x0))
    return (max(0, x0 - mx), max(0, y0 - my),
            min(image.width, x1 + mx), min(image.height, y1 + my))


def _tiles_for(model, image, window):
    """PIL crop -> EarthDial's own dynamic tiling -> (pixel_values, num_patches)."""
    import rasterio
    import torch
    from PIL import Image
    from rasterio.windows import Window
    from earthdial.train.dataset import build_transform, dynamic_preprocess

    if image.rgb8_path is None:
        raise ToolError(ErrorCode.BAD_INPUT,
                        f"{image.image_id} has no rgb8_path; model tools never read prepared_path")
    x0, y0, x1, y1 = window or (0, 0, image.width, image.height)
    with rasterio.open(image.rgb8_path) as src:
        arr = src.read(indexes=[1, 2, 3] if src.count >= 3 else [1, 1, 1],
                       window=Window(x0, y0, x1 - x0, y1 - y0))
    pil = Image.fromarray(arr.transpose(1, 2, 0).astype("uint8"))

    size = model.config.force_image_size or model.config.vision_config.image_size
    transform = build_transform(is_train=False, input_size=size, normalize_type="imagenet")
    if getattr(model.config, "dynamic_image_size", False):
        crops = dynamic_preprocess(
            pil, image_size=size,
            min_num=getattr(model.config, "min_dynamic_patch", 1),
            max_num=getattr(model.config, "max_dynamic_patch", 6),
            use_thumbnail=getattr(model.config, "use_thumbnail", False))
    else:
        crops = [pil]
    pixel_values = torch.stack([transform(c) for c in crops])
    return pixel_values, len(crops)


class RSVLM(BaseTool):
    """Inputs: image, or t1 + t2; optional region. Outputs: text, score."""

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        import torch

        p = call.params
        mode = str(p.get("mode", "vqa"))
        if mode not in prompts.MODES:
            raise ToolError(ErrorCode.BAD_INPUT, f"mode must be one of {prompts.MODES}")
        max_new_tokens = int(p.get("max_new_tokens", 64))

        model, tokenizer, model_id = _load()
        # _load() puts the EarthDial source on sys.path, so these imports follow it
        from earthdial.conversation import get_conv_template
        from earthdial.train.constants import (IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                               IMG_START_TOKEN)

        region = ctx.resolve(call.inputs["region"]) if "region" in call.inputs else None

        pair = mode == "describe_change"
        keys = ("t1", "t2") if pair else ("image",)
        missing = [k for k in keys if k not in call.inputs]
        if missing:
            raise ToolError(ErrorCode.BAD_INPUT, f"{mode} needs {keys}, missing {missing}")
        images = [ctx.resolve(call.inputs[k]) for k in keys]

        window = _crop_window(images[0], region) if region is not None else None
        parts = [_tiles_for(model, im, window) for im in images]
        pixel_values = torch.cat([pv for pv, _ in parts]).to(
            next(model.parameters()).dtype).to(next(model.parameters()).device)
        num_patches_list = [n for _, n in parts]

        benchmark = getattr(ctx.profile, "input_mode", "geotiff") == "benchmark"
        question = prompts.build(mode, str(p.get("question", "")), benchmark=benchmark)

        # --- mirrors EarthDial chat(), but generates with scores
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            query = query.replace(
                "<image>",
                f"{IMG_START_TOKEN}"f"{IMG_CONTEXT_TOKEN * model.num_image_token * num_patches}"f"{IMG_END_TOKEN}", 1)

        inputs = tokenizer(query, return_tensors="pt").to(pixel_values.device)
        try:
            out = model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens, min_new_tokens=1,
                do_sample=False, num_beams=1, eos_token_id=eos_token_id,
                output_scores=True, return_dict_in_generate=True,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise ToolError(ErrorCode.OUT_OF_MEMORY, f"rs_vlm out of memory: {exc}") from exc
            raise ToolError(ErrorCode.INTERNAL, f"rs_vlm generation failed: {exc}") from exc

        # EarthDial's generate() feeds the language model inputs_embeds (that is how
        # the image patches are spliced in), and generating from embeds returns ONLY
        # the new tokens -- there are no prompt ids to echo back. Slicing off the
        # prompt length here would discard the entire answer. Handle both shapes.
        seq = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        generated = seq[prompt_len:] if seq.shape[0] > prompt_len else seq
        text = tokenizer.decode(generated, skip_special_tokens=True).split(template.sep)[0].strip()

        score = None
        try:  # mean probability of the tokens actually chosen (master F3.5)
            probs = [
                float(torch.softmax(step_logits[0].float(), dim=-1)[token])
                for step_logits, token in zip(out.scores, generated)
            ]
            if probs:
                score = max(0.0, min(1.0, sum(probs) / len(probs)))
        except Exception as exc:
            log.warning("rs_vlm could not compute score: %s", exc)

        warnings = []
        if region is not None:
            warnings.append(f"cropped to region bbox {window} plus 10% margin")
        if not text:
            warnings.append("model returned empty text")

        return ToolResult(
            step_id=call.step_id, tool=self.card.tool, model_id=model_id,
            status="ok", text=text, score=score,
            metrics={"tiles": float(sum(num_patches_list))}, warnings=warnings,
        )
