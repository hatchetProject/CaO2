import os
import random
import warnings
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from validation.utils import (
    ImageFolder,
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
    load_model,
    DiffAugment,
    ParamDiffAug,
)


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    best_acc_l = []
    for i in range(args.repeat):
        print(f"Repeat {i+1}/{args.repeat}")
        best_acc = main_worker(args)
        best_acc_l.append(best_acc)
    print(f'\n({args.repeat} repeats) Best, last acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}')


def update_ema(model, ema_model, decay):
    model_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())
    for key in model_params.keys():
        # Perform EMA update
        ema_params[key].data.mul_(decay).add_(model_params[key].data, alpha=1 - decay)


def main_worker(args):
    print("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))
    # print(f"Refine: {args.refine}; Temperature: {args.temperature}; DSA Usage: {args.dsa}; DSA Strategy: {args.dsa_strategy}")
    teacher_model = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    )

    student_model = load_model(
        model_name=args.stud_name,
        dataset=args.subset,
        pretrained=False,
        classes=args.classes,
    )
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    student_model = torch.nn.DataParallel(student_model).cuda()

    # ema_student_model = copy.deepcopy(student_model).cuda()
    # for param in ema_student_model.parameters():
    #     param.requires_grad = False

    teacher_model.eval()
    student_model.train()

    # freeze all layers
    for param in teacher_model.parameters():
        param.requires_grad = False

    cudnn.benchmark = True

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.cos == True:
        scheduler = LambdaLR(
            optimizer,
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
            if step <= args.re_epochs
            else 0,
            last_epoch=-1,
        )
    else:
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
            last_epoch=-1,
        )

    print("process data from {}".format(args.syn_data_path))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augment = []
    augment.append(transforms.ToTensor())
    augment.append(ShufflePatches(args.factor))
    augment.append(
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)

    if args.subset == 'imagenet-woof':
        file_list = './misc/class_woof.txt'
    elif args.subset == 'imagenet-nette':
        file_list = './misc/class_nette.txt'
    elif args.subset == 'imagenet-100':
        file_list = './misc/class100.txt'
    elif args.subset == 'imagenet-1k':
        file_list = './misc/class_indices.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()
    phase = max(0, 0)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]

    train_dataset = ImageFolder(
        classes=range(args.nclass),
        ipc=args.ipc,
        mem=True,
        shuffle=True,
        root=args.syn_data_path,
        transform=transforms.Compose(augment),
        class_names=sel_classes
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.re_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(
            classes=args.classes,
            ipc=args.val_ipc,
            mem=True,
            root=args.val_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize(args.input_size // 7 * 8, antialias=True),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            class_names=sel_classes
        ),
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    print("load data successfully")

    best_acc1 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.re_epochs):
        train(epoch, train_loader, teacher_model, student_model, args)

        if epoch % 10 == 9 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.8:
                top1 = validate(student_model, args, epoch)
            else:
                top1 = 0
        else:
            top1 = 0

        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch

    print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")
    return best_acc1


def smooth_labels(targets, num_classes, smoothing=0.1):
    '''
    Args:
        targets: [N], where N is batch size
        num_classes: int, number of classes
        smoothing: float, smoothing factor
    
    Returns:
        smoothed_labels: [N, C] where C is num_classes
    '''
    assert 0 <= smoothing < 1
    with torch.no_grad():
        # Create an array of size [N, num_classes] filled with the value smoothing / num_classes
        smoothed_labels = torch.full(size=(targets.size(0), num_classes), fill_value=smoothing / num_classes).cuda()
        # Assign 1 - smoothing to the target class index
        smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return smoothed_labels


def refined_label(pred, temperature):
    """Generate refined soft labels"""
    alpha = 0.1
    pred = F.softmax(pred / temperature, dim=1)
    hard_labels = torch.argmax(pred, dim=1)
    smooth_hard_label = smooth_labels(hard_labels, num_classes=pred.shape[1], smoothing=0.1)
    refine_label = (1. - alpha) * pred + alpha * smooth_hard_label
    return refine_label


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true = F.normalize(y_true, p=2, dim=-1)
        y_pred = F.normalize(y_pred, p=2, dim=-1)
        cosine_similarity = torch.sum(y_true * y_pred, dim=-1)
        loss = 1 - cosine_similarity  # Make it a loss
        return torch.mean(loss)


def train(epoch, train_loader, teacher_model, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ema_decay = 0.99

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    loss_function_cos = CosineSimilarityLoss()
    teacher_model.eval()
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            if args.dsa:
                mix_images = DiffAugment(images, args.dsa_strategy, param=ParamDiffAug())
            else:
                mix_images, _, _, _ = mix_aug(images, args)

            if args.dsa:
                images = DiffAugment(images, args.dsa_strategy, param=ParamDiffAug())
                pred_label = student_model(images)
            else:
                pred_label = student_model(images)

            soft_mix_label = teacher_model(mix_images)
            if args.refine:
                soft_mix_label = refined_label(soft_mix_label, args.temperature)
            else:
                soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)
        
        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))

        pred_mix_label = student_model(mix_images)

        soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)

        # if args.refine:
        #     loss = loss_function_cos(soft_pred_mix_label, soft_mix_label)
        # else:
        loss = loss_function_kl(soft_pred_mix_label, soft_mix_label)

        loss = loss / args.re_accum_steps

        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        # # Update EMA model weights
        # update_ema(student_model, ema_student_model, ema_decay)

        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    # if epoch > args.re_epochs * 0.8 and (epoch + 1) % 30 == 0:
    #     printInfo = (
    #         "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
    #         + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
    #         + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
    #         + "train_time = {:.6f}".format((time.time() - t1))
    #     )
    #     print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST: Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    print(logInfo)
    logInfo = (
        "TEST: Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    return top1.avg


if __name__ == "__main__":
    pass
    # main(args)
