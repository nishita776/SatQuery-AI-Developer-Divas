"""Make EarthDial trainable under transformers 5.x. Idempotent.

Five source patches. 1-3 are also recorded in
satquery/models/rs_vlm/DECISION.md; 4-5 are training-only.

  1. config __init__ crashes when transformers calls cls() with no args
  2. .item() on a meta tensor during meta-device init
  3. their vendored Phi-3 is a 4.37 fork using DynamicCache.from_legacy_cache,
     removed in v5 -- point at the maintained implementation
  4. vision tower emits fp32 while the LLM embeddings are bf16, and forward
     index-puts one into the other
  5. cosmetic: 24 identical "Flash Attention is not available" prints. sm_90
     uses torch SDPA, which is the FlashAttention-2 algorithm anyway.
"""

import re, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "EarthDial")
D = ROOT / "src/earthdial/model/internvl_chat"

p = D / "configuration_internvl_chat.py"
s = p.read_text()
n = s.count("llm_config['architectures'][0]")
if n:
    p.write_text(s.replace("llm_config['architectures'][0]",
                           "(llm_config or {}).get('architectures', ['Phi3ForCausalLM'])[0]"))
print(f"patch 1 (config guard):       {n} replaced")

p = D / "modeling_intern_vit.py"
s = p.read_text()
pat = re.compile(r"dpr\s*=\s*\[\s*x\.item\(\)\s*for\s+x\s+in\s+torch\.linspace\([^\]]*\)\s*\]")
hit = bool(pat.search(s))
if hit:
    s = pat.sub("dpr = [config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1) "
                "for i in range(config.num_hidden_layers)]", s)
    p.write_text(s)
print(f"patch 2 (meta tensor):        {'replaced' if hit else 'already done'}")

p = D / "modeling_internvl_chat.py"
s = p.read_text()
s, n3 = re.subn(r"^from\s+\S*phi3\S*\s+import\s+.*Phi3ForCausalLM.*$",
                "from transformers import Phi3ForCausalLM", s, flags=re.M)

CAST = "vit_embeds = vit_embeds.to(input_embeds.dtype)  # patch 4: fp32 vision -> bf16 LLM"
n4 = 0
if CAST not in s:
    m = re.search(r"^(\s*)vit_embeds\s*=\s*self\.extract_feature\(pixel_values\).*$", s, re.M)
    if m:
        indent = m.group(1)
        s = s[: m.end()] + f"\n{indent}{CAST}" + s[m.end():]
        n4 = 1
p.write_text(s)
print(f"patch 3 (maintained Phi-3):   {n3} replaced")
print(f"patch 4 (vision dtype cast):  {'inserted' if n4 else 'already done'}")

n5 = 0
noisy = re.compile(r"^(\s*)print\(.*[Ff]lash.?[Aa]ttention is not (available|installed).*\)\s*$", re.M)
for name in ("modeling_intern_vit.py", "modeling_internvl_chat.py"):
    f = D / name
    t = f.read_text()
    t2, k = noisy.subn(r"\1pass  # patch 5: silenced, sm_90 uses torch SDPA", t)
    if k:
        f.write_text(t2)
    n5 += k
print(f"patch 5 (silence warnings):   {n5} prints removed")

# 6. their vendored Phi3Config validator predates the rope "su" -> "longrope"
#    rename. We set longrope so transformers' maintained Phi-3 accepts it, but
#    peft round-trips the config through this class when saving the adapter, so
#    this validator has to agree or every checkpoint fails.
p = D.parent / "phi3" / "configuration_phi3.py"
if p.exists():
    s = p.read_text()
    s2, n6 = re.subn(r"\[\s*['\"]su['\"]\s*,\s*['\"]yarn['\"]\s*\]",
                     "['su', 'yarn', 'longrope']", s)
    if n6:
        p.write_text(s2)
    print(f"patch 6 (rope name in cfg):   {n6} replaced")
else:
    print("patch 6: vendored phi3 config not found, skipped")
