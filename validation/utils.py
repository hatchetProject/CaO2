import torch
import numpy as np
import os
import torch.distributed
import torchvision
from torchvision.transforms import functional as t_F
import torch.nn.functional as F
import random
import torchvision.models as thmodels
from validation.models import ConvNet

# keep top k largest values, and smooth others
def keep_top_k(p, k, n_classes=1000):  # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes - k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(
        topk_smooth_p.sum().item(), p.shape[0]
    ), f"{topk_smooth_p.sum().item()} not close to {p.shape[0]}"
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, classes, ipc, mem=False, shuffle=False, class_names=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []
        self.targets = []
        self.samples = []
        self.all_classes = class_names
        for c in range(len(classes)):
            dir_path = os.path.join(self.root, self.all_classes[c])
            file_ls = os.listdir(dir_path)
            if shuffle:
                random.shuffle(file_ls)
            for i in range(ipc):
                self.image_paths.append(dir_path + "/" + file_ls[i])
                self.targets.append(c)
                if self.mem:
                    self.samples.append(self.loader(dir_path + "/" + file_ls[i]))

    def __getitem__(self, index):
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])
        sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None

# DSA
# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img


def load_model(model_name="resnet18", dataset="cifar10", pretrained=True, classes=[]):
    def get_model(model_name="resnet18"):
        if "conv" in model_name:
            if dataset in ["cifar10", "cifar100"]:
                size = 32
            elif dataset == "tinyimagenet":
                size = 64
            elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
                size = 128
            else:
                size = 224

            nclass = len(classes)

            model = ConvNet(
                num_classes=nclass,
                net_norm="batch",
                net_act="relu",
                net_pooling="avgpooling",
                net_depth=int(model_name[-1]),
                net_width=128,
                channel=3,
                im_size=(size, size),
            )
        elif model_name == "resnet18_modified":
            model = thmodels.__dict__["resnet18"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet101_modified":
            model = thmodels.__dict__["resnet101"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model = thmodels.__dict__[model_name](pretrained=False)

        return model

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")

        return model

    # "imagenet-100" "imagenet-10" "imagenet-first" "imagenet-nette" "imagenet-woof"
    model = get_model(model_name)
    model = pruning_classifier(model, classes)
    if pretrained:
        if dataset in [
            "imagenet-100",
            "imagenet-10",
            "imagenet-nette",
            "imagenet-woof",
            "tinyimagenet",
            "cifar10",
            "cifar100",
        ]:
            checkpoint = torch.load(
                f"./data/pretrain_models/{dataset}_{model_name}.pth", map_location="cpu"
            )
            model.load_state_dict(checkpoint["model"])
        elif dataset in ["imagenet-1k"]:
            if model_name == "efficientNet-b0":
                # Specifically, for loading the pre-trained EfficientNet model, the following modifications are made
                from torchvision.models._api import WeightsEnum
                from torch.hub import load_state_dict_from_url

                def get_state_dict(self, *args, **kwargs):
                    kwargs.pop("check_hash")
                    return load_state_dict_from_url(self.url, *args, **kwargs)

                WeightsEnum.get_state_dict = get_state_dict

            model = thmodels.__dict__[model_name](pretrained=True)

    return model