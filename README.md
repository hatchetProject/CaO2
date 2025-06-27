# CaO<sub>2</sub>
Official repository for "Rectifying Inconsistencies in Diffusion-Based Dataset Distillation". 

# Update Log
**(2025.6.26)** Starting to update code.

# Preparation
## Dataset
Download the [ImageNet](https://image-net.org/download) dataset, and place it at your intended IMAGENET_PATH.

## Installation
```
git clone https://github.com/hatchetProject/CaO2.git
cd CaO2
conda env create -f environment.yaml
conda activate cao2
```

Download the pretrained [DiT model ((DiT/XL 256×256))](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt), and place it in ``pretrained_models``.

