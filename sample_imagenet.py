import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.nn.functional as F
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
from PIL import Image
import sys
import random

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform_size = transforms.Compose([transforms.Resize((224, 224))])

    # Labels to condition the model
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

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    model_teacher = torchvision.models.__dict__[args.arch_name](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    def save_noise():
        latent_size = args.image_size // 8
        torch.manual_seed(42)
        all_noise = torch.randn(1024, 4, latent_size, latent_size)

        fname = f'noise_{args.image_size}.pt'
        if os.path.exists(fname):
            print(f"Loading noise from existing file {fname}.")
        else: 
            torch.save(all_noise, fname)
            print(f"Noise saved to {fname}.")
    save_noise()
    all_noise = torch.load(f'noise_{args.image_size}.pt')
    all_noise = all_noise.to(device)

    batch_size = args.ipc

    for c in tqdm(range(len(sel_classes))):
        model.eval()
        class_label, sel_class = class_labels[c], sel_classes[c]
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)

        new_batch_size = batch_size * 4
        # Create sampling noise:
        z = torch.randn(new_batch_size, 4, latent_size, latent_size, device=device)
        y = torch.tensor([class_label] * new_batch_size, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * new_batch_size, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        init_samples = vae.decode(samples / 0.18215).sample

        # Get the correct and most confident samples
        prob = F.softmax(model_teacher(transform_size(init_samples)), dim=1)
        pred = torch.argmax(prob, dim=1)
        correct_indices = torch.where(pred == class_label)[0]
        confidence = torch.max(prob, dim=1).values
        confident_indices = torch.argsort(confidence, descending=True)
        result_indices = torch.tensor([x for x in confident_indices if x in correct_indices])[:batch_size]
        if len(result_indices) < batch_size:
            print("Wrong samples still exists.")
            result_indices = torch.cat([result_indices, torch.tensor([x for x in confident_indices if x not in correct_indices])[:batch_size-len(result_indices)]]).long()
        samples = samples[result_indices]
        init_samples = init_samples[result_indices]

        model.train()
        ts = range(1, args.num_sampling_steps // 4)  # ADJUSTABLE      
        train_size = 1

        if not os.path.exists(os.path.join(args.save_dir, sel_class)):
            os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)

        l_inf_norm_list = []
        total_samples = None
        lambda_linf = 10.0 # ADJUSTABLE
        for img_idx, x0 in enumerate(samples):
            # Setting class label as predicted by teacher model
            with torch.enable_grad():
                latent_ = x0.detach().clone().unsqueeze(0)
                latent_ = torch.nn.Parameter(latent_, requires_grad=True)
                optimizer = torch.optim.Adam([latent_], lr=6e-4)
                noise = all_noise[np.random.choice(np.arange(all_noise.size(0)), train_size)]
                for _ in range(100):
                    optimizer.zero_grad()
                    batch_ts = torch.tensor(np.random.choice(ts, train_size), device=device)
                    # predicted_noise = pipe.unet(latents, timestep, encoder_hidden_states=text_embeddings)["sample"]
                    noised_latent = diffusion.q_sample(latent_.repeat(train_size, 1, 1, 1), batch_ts, noise)
                    ## True class label
                    # uncond_input_label = torch.tensor([class_label] * len(batch_ts), device=device)
                    ## Null class label
                    uncond_input_label = torch.tensor([1000] * len(batch_ts), device=device)

                    model_output_label = model(noised_latent, batch_ts, y=uncond_input_label)
                    B, C = noised_latent.shape[:2]
                    noise_label, _ = torch.split(model_output_label, C, dim=1)
                    linf_norm = torch.max(torch.abs(noise - noise_label))
                    mse_error = F.mse_loss(noise, noise_label, reduction='none').mean(dim=(0, 1, 2, 3))

                    loss = mse_error + lambda_linf * linf_norm
                    loss.backward()
                    optimizer.step()

            latent_ = latent_.detach().data
            save_sample = vae.decode(latent_ / 0.18215).sample
            # Save and display images:
            save_image(save_sample, os.path.join(args.save_dir, sel_class, f"{img_idx}.png"), normalize=True, value_range=(-1, 1))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2.7)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--ipc", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')

    parser.add_argument("--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models")

    parser.add_argument('--t_interval', type=int, default=4, help='Timestep interval')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    args = parser.parse_args()

    main(args)
    
