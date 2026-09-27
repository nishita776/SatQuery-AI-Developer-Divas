#!/usr/bin/env bash
# Full LAE-DINO recovery. Run after a box wipe. Creates env 'lae'.
set -e
cd ~
[ -d LAE-DINO ] || git clone https://github.com/jaychempan/LAE-DINO.git
conda env list | grep -q "^lae " || conda create -n lae python=3.10 -y
source activate lae
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmengine "numpy<2" "setuptools<70"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
cd ~/LAE-DINO/mmdetection_lae && pip install -v -e . --no-build-isolation
pip install "transformers==4.36.2" "opencv-python-headless<5" "numpy<2" \
  ftfy regex open_clip_torch fairscale yapf pycocotools terminaltables \
  prettytable nltk fvcore timm scipy shapely emoji ddd-dataset
pip install git+https://github.com/openai/CLIP.git || pip install clip-anytorch
# weights
cd ~/LAE-DINO
hf download ML4Sustain/LAE-DINO checkpoints/lae_dino_swint_lae1m-28ca3a15.pth --local-dir weights/
hf download google-bert/bert-base-uncased --local-dir weights/bert-base-uncased
# nltk data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
# patch vendored det_inferencer for multi-dataset concat config
cd ~/LAE-DINO/mmdetection_lae
python - <<'PY'
f="mmdet/apis/det_inferencer.py"; s=open(f).read()
o1="        metainfo = DATASETS.build(test_dataset_cfg).metainfo\n"
n1=o1+"        if isinstance(metainfo, list):\n            metainfo = metainfo[0] if metainfo else {}\n"
if o1 in s and "isinstance(metainfo, list)" not in s: s=s.replace(o1,n1)
o2="        pipeline_cfg = cfg.test_dataloader.dataset.pipeline\n"
n2=("        _ds = cfg.test_dataloader.dataset\n"
    "        while 'pipeline' not in _ds and 'datasets' in _ds:\n"
    "            _ds = _ds['datasets'][0]\n"
    "        pipeline_cfg = _ds.pipeline\n")
if o2 in s: s=s.replace(o2,n2)
open(f,"w").write(s); print("det_inferencer patched")
PY
# dummy dataset jsons the benchmark config demands
python - <<'PY'
from mmengine.config import Config
import os, json
cfg=Config.fromfile('configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py')
d={"images":[{"id":1,"file_name":"d.jpg","height":10,"width":10}],
"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[0,0,1,1],"area":1,"iscrowd":0}],
"categories":[{"id":1,"name":"object"}]}
def walk(o):
    if isinstance(o,dict):
        if 'ann_file' in o and isinstance(o['ann_file'],str):
            p=os.path.join(o.get('data_root','') or '',o['ann_file'])
            os.makedirs(os.path.dirname(p) or '.',exist_ok=True)
            if not os.path.exists(p): json.dump(d,open(p,'w'))
        for v in o.values(): walk(v)
    elif isinstance(o,(list,tuple)):
        [walk(x) for x in o]
walk(cfg.test_dataloader); print("dummy jsons ready")
PY
echo "LAE-DINO READY. Count-off:"
echo "  conda activate lae && cd ~/LAE-DINO/mmdetection_lae"
echo "  python demo/image_demo.py --inputs <img.png> --model configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py --weights ~/LAE-DINO/weights/checkpoints/lae_dino_swint_lae1m-28ca3a15.pth --texts 'building . vehicle . road' --pred-score-thr 0.4 --out-dir outputs --print-result"
