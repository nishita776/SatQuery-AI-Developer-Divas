# ML-01: RS-VLM backbone decision

**Decided 2026-09-20 (Day 0). Locked — no switching after Day 0 per handover.**

## Chosen: EarthDial 4B RGB

| | |
|---|---|
| Weights | `akshaydudhane/EarthDial_4B_RGB` (HuggingFace, public) |
| Code | https://github.com/hiyamdebary/EarthDial (CVPR 2025, arXiv 2412.15190) |
| Architecture | InternVL2: `InternVisionModel` + Phi-3 Mini (`Phi3ForCausalLM`), 4B |
| Variants available | `_4B_RGB`, `_4B_MS`, `_4B_Methane_UHI` |
| Input | 448x448 per tile, `force_image_size: 448`, `downsample_ratio: 0.5` |
| Dynamic tiling | `max_dynamic_patch: 6` (+ thumbnail) — built into the model |
| Chat template | `phi3-chat` |
| Load method | 4-bit NF4, `BitsAndBytesConfig`, compute dtype **fp16**, `device_map={"": 0}` |
| **GPU memory** | **2.15 GiB** allocated on a Tesla T4 |
| **Load time** | **6.6-7.5 s** warm; first download 8.29 GB (~37 s) |
| Sample Q/A | "Describe this satellite image." -> "1 large airplane" (random-noise placeholder input) |

Verified environment: Kaggle, Python 3.12.13, transformers 5.0.0,
torch 2.10.0+cu128, bitsandbytes 0.50.2, Tesla T4 (sm_75).

## Why not GeoChat-7B

GeoChat pins `transformers==4.31.0` and `torch==2.0.1` — six minor versions
further back than EarthDial's 4.37.2, and a torch pin that fights a CUDA 12.8
driver. EarthDial is also 4B rather than 7B, which leaves ~12 GiB of T4
headroom for ML-08's QLoRA. GeoChat's input resolution, recorded for
completeness, is 504x504 (CLIP ViT-L/14-336 interpolated, 1296 patches).

## [to verify] items resolved

- **Are EarthDial's code and weights public?** Yes, both. The HF repo ships
  weights + `inference.py` but **not** the modelling code — `auto_map` points at
  `configuration_internvl_chat.py`, which is not uploaded. `trust_remote_code`
  therefore fails; you must clone the GitHub repo and put `src/` on `sys.path`.
- **GeoChat input resolution?** 504x504.
- **(ML-04) `post_process_grounded_object_detection` arg names?** In
  transformers 5.x the method is on `GroundingDinoProcessor` with signature
  `(outputs, input_ids=None, threshold=0.25, text_threshold=0.25,
  target_sizes=None, text_labels=None)`. The box arg is **`threshold`**, not
  `box_threshold`. Results carry `scores`, `boxes`, `labels` (integer IDs since
  4.51) and `text_labels` — use **`text_labels`** for `Detection.label`.

## Compatibility shims required on transformers 5.0.0

EarthDial targets `transformers==4.37.2`. All of the following are needed to run
it on the stock Kaggle image. **ML-06 must reproduce these inside `run_real()`.**

### File patches (after cloning the repo)

In `src/earthdial/model/internvl_chat/`:

1. `configuration_internvl_chat.py` — replace all 5 occurrences of
   `llm_config['architectures'][0]` with
   `(llm_config or {}).get('architectures', ['Phi3ForCausalLM'])[0]`.
   v5's `to_diff_dict()` instantiates the config class with no arguments;
   the original raises `KeyError: 'architectures'`.

2. `modeling_intern_vit.py` (~line 245) — replace
   `dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]`
   with
   `dpr = [config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1) for i in range(config.num_hidden_layers)]`.
   v5 builds models under a meta-device context; `.item()` is illegal on meta
   tensors.

3. `modeling_internvl_chat.py` (line 16) — replace
   `from earthdial.model.phi3.modeling_phi3 import Phi3ForCausalLM` with
   `from transformers import Phi3ForCausalLM`.
   **This is the important one.** Their vendored Phi-3 is a fork of
   transformers 4.37 that calls `DynamicCache.from_legacy_cache`, removed in v5.
   Switching to the maintained implementation eliminates the entire
   cache-and-generation failure class, and makes `GenerationMixin` and
   `generation_config` work natively with no further patching.

### Runtime shims

4. `InternVLChatModel.all_tied_weights_keys = {}` — v5's bnb quantizer reads
   this attribute during `_process_model_before_weight_loading`.

5. On the loaded config, rewrite the rope type:
   `cfg.llm_config.rope_scaling["type"] = cfg.llm_config.rope_scaling["rope_type"] = "longrope"`
   (it ships as `"su"`) and ensure
   `cfg.llm_config.original_max_position_embeddings = 4096`.
   transformers renamed the Phi-3 rope type; `ROPE_INIT_FUNCTIONS` has no `su`.

6. Wrap `model.language_model.generate` to drop a `return_dict` kwarg.
   EarthDial's `generate()` forwards `return_dict=`, and v5's `_prefill`
   supplies it again when calling the forward -> duplicate keyword argument.

7. `earthdial.train.dataset` imports `decord` (a video library) at module level.
   It has no Python 3.12 wheel. Stub it in `sys.modules` before importing
   `build_transform`, or reimplement the transform: bicubic resize to 448,
   `ToTensor`, ImageNet mean/std normalisation.

### Call-site flags

8. `torch_dtype=torch.float16` — **not** `bfloat16` as their `inference.py` uses.
   T4 is sm_75; `torch.cuda.is_bf16_supported()` returns True but that counts
   emulation. `is_bf16_supported(including_emulation=False)` is False.
   This is also why ML-08 trains the LoRA in fp16.
9. Pass `quantization_config=BitsAndBytesConfig(...)`, not `load_in_4bit=True`
   (bare kwarg removed in v5).
10. Do **not** pass `use_flash_attn` to `from_pretrained` — EarthDial's
    `__init__` does not accept it. Flash-attn 2 needs sm_80+ anyway and is
    absent on Kaggle; the model falls back on its own.

## Open issues for ML-06

- **Decoding artifact.** Output arrives as `1 ▁large ▁air plane` — raw
  SentencePiece pieces rather than decoded text. `rs_vlm` must return clean
  `text`. First thing to try: `use_fast=True` on the tokenizer.
- **Double tiling.** EarthDial does its own dynamic tiling (up to 6 crops of
  448 px). The Scale Bridge (ML-03) also tiles. Decide deliberately: probably
  hand `rs_vlm` a region crop and let the model tile internally, rather than
  tiling twice.
- **Environment split.** Not required. EarthDial runs on transformers 5.0.0
  with the shims above, so the detector (Grounding DINO, needs >=4.40) can share
  one Kaggle environment. No `transformers<5` pin is needed in `pyproject.toml`.
