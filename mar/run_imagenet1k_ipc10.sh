#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

IMAGENET_PATH=your path
IMAGENET_TRAIN_PATH=your path
IMAGENET_VAL_PATH=your path

# Recover MAR-Base
python robust_imagenet1k.py --arch-name "resnet18" --spec imagenet-1k --exp-name "1k_rn18_ipc10" \
    --syn-data-path 'results' --batch-size 10 --lr 0.001 --r-bn 0.01 --iteration 0 \
    --store-best-images --easy2hard-mode "cosine" --milestone 1 --ipc-start 0 --ipc-end 10 \
    --model mar_base --resume pretrained_models/mar/mar_base --diffloss_d 6 --diffloss_w 1024 \
    --num_iter 32 --num_sampling_steps 100 --cfg 2.9 --cfg_schedule linear --temperature 1.0 --class_num 1000


python main_validate.py --subset "imagenet-1k" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc 10 --stud-name "resnet101" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path results/1k_rn18_ipc10
