# CaO<sub>2</sub>
Official repository for [CaO<sub>2</sub>: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](https://arxiv.org/abs/2506.22637v1). 

# Update Log
**(2025.7.7)** Upload distilled datasets.

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

Download the pretrained classification models for RDED evaluation from [RDED](https://github.com/LINs-lab/RDED) and place them in ``data/pretrain_models``. We also supplement the pretrained models (ResNet50 and ResNet101 for Nette) [here](https://drive.google.com/file/d/1TSI6VemPz3337qV0Cbre6pZhNnM-SvvL/view?usp=sharing). The script for running evaluation without teacher supervision is in ``eval.py``. Though more reasonable, its performance is inferior to those that use teacher-student training.

# Usage
We provided an example script in ``run.sh`` for the users to run the code. Note that the users will need to modify the variables in the script for it to run properly. 

To gain better performance, hyperparameters can be tuned, but the variations are usually marginal.

# Distilled Datasets
For direct evaluation and benchmarking, the distilled datasets are uploaded [here](https://drive.google.com/file/d/1Nyk67RrjPirnjaInxLuID4csb5teN4eD/view?usp=sharing) (>7GB).

# TODO
- [x] Complete the code upload
- [x] Upload the distilled datasets
- [x] Update code for MAR distillation

# Acknowledgements
Thanks to these amazing repositories: [Minimax Diffusion](https://github.com/vimar-gu/MinimaxDiffusion), [RDED](https://github.com/LINs-lab/RDED) and many other inspiring works.

# Citation
If you find this work useful, please consider citing:
```
@misc{wang2025cao2rectifyinginconsistenciesdiffusionbased,
      title={CaO$_2$: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation}, 
      author={Haoxuan Wang and Zhenghao Zhao and Junyi Wu and Yuzhang Shang and Gaowen Liu and Yan Yan},
      year={2025},
      eprint={2506.22637},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.22637}, 
}
```
