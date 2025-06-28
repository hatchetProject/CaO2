# CaO<sub>2</sub>
Official repository for "Rectifying Inconsistencies in Diffusion-Based Dataset Distillation". 

# Update Log
**(2025.6.26)** Starting to update the repository.

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

Download the pretrained [DiT model (DiT/XL 256×256)](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt), and place it in ``pretrained_models``.

Download the pretrained classification models for RDED evaluation from [RDED](https://github.com/LINs-lab/RDED) and place them in ``data/pretrain_models``. We also provide the script for running evaluation without teacher supervision.

# Usage
We provided an example script in ``run.sh`` for the users to run the code. Note that the users will need to modify the variables in the script for it to run properly.

# TODO
- [ ] Complete the code upload
- [ ] Upload the distilled datasets

# Acknowledgements
Thanks to these amazing repositories: [Minimax Diffusion](https://github.com/vimar-gu/MinimaxDiffusion), [RDED](https://github.com/LINs-lab/RDED) and many other inspiring works.

# Citation
If you find this work useful, please consider citing:
