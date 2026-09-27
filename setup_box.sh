#!/usr/bin/env bash
set -e
cd ~
[ -d SatQuery-AI-Developer-Divas ] || git clone https://github.com/nishita776/SatQuery-AI-Developer-Divas.git
[ -d EarthDial ] || git clone https://github.com/hiyamdebary/EarthDial.git
cd ~/EarthDial && pip install -e . -q
pip install -q einops timm accelerate sentencepiece tiktoken protobuf \
  opencv-python-headless datasets pyarrow huggingface_hub peft bitsandbytes \
  matplotlib decord yacs addict termcolor scipy scikit-image
python ~/SatQuery-AI-Developer-Divas/training/patch_earthdial.py ~/EarthDial
cd ~/SatQuery-AI-Developer-Divas && pip install -e ".[geo,dev]" -q
export CC=/opt/conda/bin/gcc CXX=/opt/conda/bin/g++
echo "DONE. Adapter: hf download 19purple/satquery-lora --local-dir training/adapters/lora-v1"
