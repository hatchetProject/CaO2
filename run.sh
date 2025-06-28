export CUDA_VISIBLE_DEVICES=0

IMAGENET_PATH=Set your path here
IMAGENET_TRAIN_PATH=Set your path here
IMAGENET_VAL_PATH=Set your path here


IPC=10
# Datasets including: imagenet-woof (N_CLASS=10), imagenet-nette (N_CLASS=10), imagenet-100 (N_CLASS=100), imagenet-1k (N_CLASS=1000)
SPEC=imagenet1k
N_CLASS=1000

# run sample generation
python sample_imagenet.py --model DiT-XL/2 --arch-name "resnet18" --image-size 256 --ipc IPC --save-dir results/SPEC_IPC --spec SPEC --nclass 1000

# run evaluation: the following is for RDED evaluation process, which reproduces the paper results
# Evaluation model type: resnet18, resnet50, resnet101
python main_validate.py --subset "SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc IPC --stud-name "resnet18" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path results/SPEC_IPC

python main_validate.py --subset "SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc IPC --stud-name "resnet50" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path results/SPEC_IPC

python main_validate.py --subset "SPEC" --arch-name "resnet18" --factor 2 \
    --num-crop 5 --mipc 300 --ipc IPC --stud-name "resnet101" --re-epochs 300 \
    --train-dir $IMAGENET_TRAIN_PATH --val-dir $IMAGENET_VAL_PATH --repeat 3 \
    --syn-data-path results/SPEC_IPC


# Regular evaluation, this example evaluates on resnet18
python eval.py -d imagenet --imagenet_dir results/SPEC_IPC $IMAGENET_PATH \
     -n resnet --depth 18 --nclass N_CLASS --norm_type instance --ipc IPC --tag test --slct_type random --spec SPEC --repeat 3
