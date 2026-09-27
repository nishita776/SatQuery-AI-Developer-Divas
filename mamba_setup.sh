#!/usr/bin/env bash
# Full MambaSCD (ChangeMamba) recovery. Run after a box wipe. Creates env 'mamba'.
set -e
cd ~
[ -d ChangeMamba ] || git clone https://github.com/ChenHongruixuan/ChangeMamba.git
conda env list | grep -q "^mamba " || conda create -n mamba python=3.10 -y
source activate mamba
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# pod ships CUDA 13.1 which won't compile against torch cu121 -> force 12.1 toolchain + gcc<=12
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64
export CUDA_HOME=$CONDA_PREFIX
pip install "numpy<2" "setuptools<70" wheel ninja packaging
pip install causal-conv1d==1.1.1 --no-build-isolation
cd ~/ChangeMamba/kernels/selective_scan && rm -rf build && pip install . --no-build-isolation
pip install imageio scikit-image scikit-learn "opencv-python-headless<5" tqdm tensorboard \
  matplotlib pandas easydict pyyaml termcolor addict yacs timm einops fvcore ml_collections \
  thop tensorboardX seaborn h5py
pip install "numpy<2"
cd ~/ChangeMamba && mkdir -p saved_models data
[ -f saved_models/MambaSCD_Tiny_SECOND.pth ] || curl -L -o saved_models/MambaSCD_Tiny_SECOND.pth "https://zenodo.org/records/15479555/files/MambaSCD_Tiny_SECOND_SeK_0.2208.pth?download=1"
[ -f saved_models/vssm_tiny.pth ] || curl -L -o saved_models/vssm_tiny.pth "https://zenodo.org/records/15479555/files/vssm_tiny_0230_ckpt_epoch_262.pth?download=1"
[ -d data/SECOND ] || { curl -L -o data/SECOND.zip "https://zenodo.org/records/15479555/files/SECOND.zip?download=1"; cd data && unzip -q SECOND.zip && cd ~/ChangeMamba; }
echo "MambaSCD READY. Infer:"
echo "  conda activate mamba && cd ~/ChangeMamba && export CUDA_HOME=\$CONDA_PREFIX"
echo "  python changedetection/script/infer_MambaSCD.py --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml --model_type MambaSCD_Tiny --model_checkpoint_path saved_models/MambaSCD_Tiny_SECOND.pth --encoder_pretrained_path saved_models/vssm_tiny.pth --dataset SECOND --test_dataset_path data/SECOND/test --test_data_list_path data/SECOND/test.txt --result_saved_path ~/ChangeMamba/results"
