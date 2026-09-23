"""Prompt templates for rs_vlm's four modes (master F4.3).

Imported by BOTH training/prepare.py and rs_vlm.run_real(). Single source of
truth on purpose: if the phrasing the LoRA is trained on drifts from the
phrasing sent at inference, the adapter degrades silently.
"""

from __future__ import annotations

IMAGE = "<image>"

CAPTION = "Describe this satellite image."
DESCRIBE_SCENE = (
    "Describe the scene in this satellite image, including the land cover "
    "and any notable structures."
)
DESCRIBE_CHANGE = "Describe what has changed between the two images."
BENCHMARK_SUFFIX = "\nAnswer with one word or a number."

MODES = ("vqa", "caption", "describe_change", "describe_scene")


def single(body: str) -> str:
    """One image (F4.3: vqa, caption, describe_scene)."""
    return f"{IMAGE}\n{body}".strip()


def pair(body: str) -> str:
    """Two images (F4.3: describe_change, and CHANGE_VQA)."""
    return f"Image 1: {IMAGE}\nImage 2: {IMAGE}\n{body}".strip()


def build(mode: str, question: str = "", benchmark: bool = False) -> str:
    """The exact user turn for a mode. Raises on an unknown mode."""
    q = (question or "").strip()
    if mode == "vqa":
        body = q or CAPTION
        if benchmark:
            body += BENCHMARK_SUFFIX
        return single(body)
    if mode == "caption":
        return single(CAPTION)
    if mode == "describe_scene":
        return single(DESCRIBE_SCENE)
    if mode == "describe_change":
        return pair(q or DESCRIBE_CHANGE)
    if mode == "change_vqa":  # training-only task name
        body = q or DESCRIBE_CHANGE
        if benchmark:
            body += BENCHMARK_SUFFIX
        return pair(body)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
