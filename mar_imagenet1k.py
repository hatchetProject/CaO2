import argparse
import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb
from PIL import Image
from utils import BNFeatureHook, clip, lr_cosine_policy
import copy
from models.vae import AutoencoderKL
from models_dd import mar
import sys
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from rded_utils import mix_images, selector, denormalize, MultiRandomCrop, selector_el2n
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class BufferDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory path to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths = sorted([os.path.join(directory, fname) for fname in os.listdir(directory)],
                                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

def el2n_score(pred, y):
    with torch.no_grad():
        pred = F.softmax(pred, dim=1)
        l2_loss = torch.nn.MSELoss(reduction='none')
        y_hot = F.one_hot(y, num_classes=pred.shape[1])
        el2n = torch.sqrt(l2_loss(y_hot, pred).sum(dim=1))
    return el2n.cpu().numpy()

def get_classwise_images(args, model_teacher, hook_for_display, model, vae):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_size = transforms.Compose([transforms.Resize((224, 224))])
    # setup target labels
    ipc = args.ipc_end - args.ipc_start
    factor = 2
    for_selector = False
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'imagenet100':
        file_list = './misc/class100.txt'
    else:
        file_list = './misc/class_indices.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    current_time = datetime.now()
    buffer_name = "syn-data/buffer/" + current_time.strftime("%Y%m%d_%H%M%S")

    for c in tqdm(range(len(sel_classes))):
        model.eval()
        class_label, sel_class = class_labels[c], sel_classes[c]
        os.makedirs(os.path.join(args.syn_data_path, sel_class), exist_ok=True)

        batch_size = ipc * 2
        targets = torch.Tensor([class_label] * batch_size).long().cuda()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                ## Sample with step similarity
                start_step = args.num_iter - 2
                end_step = args.num_iter - 1
                seq_len = model.seq_len
                mask = torch.ones(batch_size, seq_len).cuda()
                tokens = torch.zeros(batch_size, seq_len, model.token_embed_dim).cuda() # [bsz, 256, 16]
                orders = model.sample_orders(batch_size)
                for step in range(args.num_iter):
                    tokens, mask = model.sample_token_single_step(tokens, mask, orders, step, bsz=batch_size, num_iter=args.num_iter,
                                                cfg_schedule=args.cfg_schedule, cfg=args.cfg, labels=targets, temperature=args.temperature)
                    if step == start_step:
                        start_step_tokens = model.unpatchify(tokens)
                        start_step_images = vae.decode(start_step_tokens / 0.2325)
                        save_buffer(buffer_name+"_start", start_step_images)
                        dataset = BufferDataset(buffer_name+"_start", transform=transform)
                        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        for image in data_loader:
                            start_outputs = F.softmax(model_teacher(image), dim=1)
                    elif step == end_step:
                        end_step_tokens = model.unpatchify(tokens)
                        end_step_images = vae.decode(end_step_tokens / 0.2325)
                        save_buffer(buffer_name+"_end", end_step_images)
                        dataset = BufferDataset(buffer_name+"_end", transform=transform)
                        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        for image in data_loader:
                            end_outputs = F.softmax(model_teacher(image), dim=1)
                cos_score = F.cosine_similarity(start_outputs, end_outputs, dim=1).cpu().numpy()
                easy_indices = np.argsort(cos_score)[-ipc:]
                sampled_tokens = tokens[easy_indices]
        
        lambda_linf = 10 # ADJUSTABLE
        mask_max_ratio = 0.25 # ADJUSTABLE
        for img_idx, x0 in enumerate(sampled_tokens):
            with torch.enable_grad():
                latent_ = x0.detach().clone().unsqueeze(0)
                latent_ = torch.nn.Parameter(latent_, requires_grad=True)
                optimizer = torch.optim.Adam([latent_], lr=6e-4)
                for _ in range(100):
                    optimizer.zero_grad()
                    # Guide with zero class embedding
                    labels = torch.zeros(1, 1000).cuda()
                    class_embedding = model.embed_linear(labels)

                    gt_latents = latent_.clone().detach()
                    orders = model.sample_orders(bsz=latent_.size(0))
                    mask = model.dd_random_masking(latent_, orders, max_ratio=mask_max_ratio)
                    encode_x = model.forward_mae_encoder(latent_, mask, class_embedding)
                    z = model.forward_mae_decoder(encode_x, mask)
                    loss_dict = model.forward_loss(z=z, target=gt_latents, mask=mask)
                    mse_error = loss_dict['loss']
                    if mask is not None:
                        mse_error = (mse_error * mask).sum() / mask.sum()
                    mse_error = mse_error.mean()
                    
                    linf_norm = torch.max(torch.abs(loss_dict["target"] - loss_dict["model_output"]))
                    loss = mse_error + lambda_linf * linf_norm

                    loss.backward()
                    optimizer.step()
        
            latent_ = latent_.detach().data
            sampled_tokens = model.unpatchify(latent_).float()
            samples = vae.decode(sampled_tokens / 0.2325)
            save_image(samples, os.path.join(args.syn_data_path, sel_class, f"{img_idx}.png"), normalize=True, value_range=(-1, 1))

        torch.cuda.empty_cache()
    

def save_images(args, images, targets, for_selector=False, all_classes=None):
    if not os.path.exists(args.syn_data_path):
        os.mkdir(args.syn_data_path)
    # save_path = "{}/new{:03d}".format(args.syn_data_path, targets[0].item())
    save_path = "{}/{}".format(args.syn_data_path, all_classes[targets[0].item()])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image_index, image in enumerate(images):
        image_np = image.data.cpu().numpy().transpose((1, 2, 0))
        if not for_selector:
            image_np = (image_np + 1) / 2 
            image_np = np.clip(image_np, 0, 1)
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(os.path.join(save_path, f"{image_index}.jpg"))
        # save_image(image, os.path.join(save_path, f"{image_index+10}.jpg"), normalize=True, value_range=(-1, 1))


def save_buffer(path, images):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, img in enumerate(images):
        save_image(img, os.path.join(path, f"{i}.jpg"), normalize=True, value_range=(-1, 1))


def validate(input, target, model):
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

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        loss = nn.CrossEntropyLoss()(output, target)

    print("Verifier accuracy: ", prec1.item())
    return prec1.item(), loss.item()

 
def main_syn(args, mar, vae):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    model_teacher = models.__dict__[args.arch_name](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    if args.verifier:
        model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
        model_verifier = model_verifier.cuda()
        model_verifier.eval()
        for p in model_verifier.parameters():
            p.requires_grad = False

        hook_for_display = lambda x, y: validate(x, y, model_verifier)
    else:
        hook_for_display = None

    get_classwise_images(args, model_teacher, hook_for_display, mar, vae)


def parse_args():
    parser = argparse.ArgumentParser("CDA for ImageNet-1K")
    """Data save flags"""
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment, subfolder under syn_data_path")
    parser.add_argument("--syn-data-path", type=str, default="./syn-data", help="where to store synthetic data")
    parser.add_argument("--store-best-images", action="store_true", help="whether to store best images")
    """Optimization related flags"""
    parser.add_argument("--batch-size", type=int, default=100, help="number of images to optimize at the same time")
    parser.add_argument("--iteration", type=int, default=1000, help="num of iterations to optimize the synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--jitter", default=32, type=int, help="random shift on the synthetic data")
    parser.add_argument("--r-bn", type=float, default=0.05, help="coefficient for BN feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0, help="additional multiplier on first bn layer of R_bn")
    """Model related flags"""
    parser.add_argument("--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models")
    parser.add_argument("--verifier", action="store_true", help="whether to evaluate synthetic data with another model")
    parser.add_argument("--verifier-arch", type=str, default="mobilenet_v2", help="arch name from torchvision models to act as a verifier")
    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)
    parser.add_argument("--G", default="-1", type=str)
    parser.add_argument("--ipc-start", default=0, type=int)
    parser.add_argument("--ipc-end", default=1, type=int)
    parser.add_argument("--wandb-key", default="", type=str)

    ## MAR parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--seed', default=1, type=int)
    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int, help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str, help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int, help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int, help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int, help='number of tokens to group as a patch.')
    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int, help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int, help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--grad_checkpointing', action='store_true')
    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7, help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0, help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1, help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)
    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    parser.add_argument("--opt", action="store_true", help="whether to optimize images")
    parser.add_argument('--class_num', default=1000, type=int, help="class number")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')

    args = parser.parse_args()

    assert args.milestone >= 0 and args.milestone <= 1

    if args.G != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.G
        print("set CUDA_VISIBLE_DEVICES to ", args.G)

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


if __name__ == "__main__":
    args = parse_args()

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # define the vae and mar model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=1000,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    device = torch.device("cuda")
    model.to(device)
    model_without_ddp = model

    # resume training
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)
        del checkpoint
    torch.cuda.empty_cache()
    model_without_ddp = model_without_ddp.cuda()
    model_without_ddp.eval()
    # Use EMA model
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = ema_params[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # Convert to soft label version
    model_without_ddp.embed_linear = torch.nn.Linear(768, 1000, bias=False)
    model_without_ddp.embed_linear.weight.data = model_without_ddp.class_emb.weight.data.T
    
    model_without_ddp.eval()
    
    main_syn(args, model_without_ddp, vae)
