from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.jit import Final
from timm.models.vision_transformer import LayerScale, Mlp, DropPath, use_fused_attn
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
from models_dd.diffloss import DiffLoss
import time
import torch.nn.functional as F

class Block(nn.Module):
    """
    timm.version 0.9.12
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Attention(nn.Module):
    """
    timm.version 0.9.12
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def dd_random_masking(self, x, orders, max_ratio=0.25):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = max(int(np.ceil(seq_len * mask_rate)), int(seq_len * max_ratio))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask


    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1) # self.buffer_size = 64, [bsz, 64+256, 768]
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1) # class_embedding: [bsz, 768] -> [bsz, 1, 768]

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        # Changed to enable DD optimization
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss_dict = self.diffloss(z=z, target=target, mask=mask)
        return loss_dict

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda() # [bsz, 256, 16]
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                if labels.dim() == 2 and labels.shape[1] == 1000:
                    class_embedding = self.embed_linear(labels)
                else:
                    class_embedding = self.class_emb(labels) # [bsz, 768]
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0) # [2 * bsz, 768]
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding) # x: [2*bsz, 64, 768]

            # mae decoder
            z = self.forward_mae_decoder(x, mask) #[2*bsz, seq_len, 768]

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next # [bsz, seq_len]
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)] # from [2*bsz, seq_len, 768] to [2*bsz, 768], meaning only takes one token
            # # A similarity here with class embedding
            # sim = similarity(z.cpu().detach().numpy(), class_embedding.cpu().detach().numpy(), t="cosine")
            # target_tokens = mask_to_pred.nonzero(as_tuple=True)
            # replace_idx = sim < 0
            # replace_idx = replace_idx[:bsz]
            # bs_idx = target_tokens[0].cpu().numpy()[:bsz][replace_idx]
            # tgt_idx = target_tokens[1].cpu().numpy()[:bsz][replace_idx]
            # for b, d in zip(bs_idx, tgt_idx):
            #     bs_list[b].append(d)
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            # Diffusion sampling
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter) # [2*bsz, 16]
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            # if 16 < step <= 18:
            #     cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = torch.zeros_like(sampled_token_latent)
            #     random_indices = np.random.randint(step-2, step, size=bsz)
            #     idx = orders[np.arange(bsz), random_indices]
            #     cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = tokens[torch.arange(bsz), idx, :]
            # else:
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone() # [bsz, 256, 16]

            # if step > 29:
            #     left_orders = orders[:, (self.seq_len // num_iter) * step:]
            #     # left_tokens = tokens[torch.arange(bsz)[:, None], left_orders].cuda()
            #     break

        # discard_indices, retain_indices = diversity_loss(tokens, n=16)
        # tokens[torch.arange(bsz)[:, None], retain_indices] = 0
        # tokens[torch.arange(bsz)[:, None], discard_indices] = tokens[torch.arange(bsz)[:, None], retain_indices]
        # for b in range(bsz):
        #     tokens[b, bs_list[b]] = torch.zeros(len(bs_list[b]), self.token_embed_dim).cuda()
        # unpatchify
        # tokens = self.unpatchify(tokens) # [bsz, 16, 16, 16]
        return tokens


    def sample_token_single_step(self, tokens, mask, orders, step, bsz, num_iter, cfg_schedule="linear", cfg=1.0, labels=None, temperature=1.0):
        cur_tokens = tokens.detach().clone()

        # class embedding and CFG
        if labels is not None:
            if labels.dim() == 2 and labels.shape[1] == 1000:
                class_embedding = self.embed_linear(labels)
            else:
                class_embedding = self.class_emb(labels) # [bsz, 768]
        else:
            class_embedding = self.fake_latent.repeat(bsz, 1)
            
        if not cfg == 1.0:
            cur_tokens = torch.cat([cur_tokens, cur_tokens], dim=0)
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0) # [2 * bsz, 768]
            mask = torch.cat([mask, mask], dim=0)
       
        # mae encoder
        x = self.forward_mae_encoder(cur_tokens, mask, class_embedding) # x: [2*bsz, 64, 768]

        # mae decoder
        z = self.forward_mae_decoder(x, mask) #[2*bsz, seq_len, 768]

        # mask ratio for the next round, following MaskGIT and MAGE.
        mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                    torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
        if step >= num_iter - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
        mask = mask_next # [bsz, seq_len]
        if not cfg == 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        # sample token latents for this step
        z = z[mask_to_pred.nonzero(as_tuple=True)] # from [2*bsz, seq_len, 768] to [2*bsz, 768], meaning only takes one token

        # cfg schedule follow Muse
        if cfg_schedule == "linear":
            cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
        elif cfg_schedule == "constant":
            cfg_iter = cfg
        else:
            raise NotImplementedError
        # Diffusion sampling
        sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter) # [2*bsz, 16]
        if not cfg == 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        tokens.data, _ = cur_tokens.clone().chunk(2, dim=0) # [bsz, 256, 16]
        return tokens, mask


def diversity_loss(tokens, n=16):
    x = tokens.detach().cpu().numpy()
    bsz, seq_len, token_dim = x.shape # [bsz, 256, 16]
    div_scores = np.empty((bsz, seq_len))
    for b in range(bsz):
        for i in range(seq_len):
            div_scores[b][i] = np.linalg.norm(x[b] - np.matmul(np.ones((seq_len, 1)), x[b][i].reshape(1, -1)), ord=1)
    smallest_n_indices = np.argsort(div_scores, axis=1)[:, :n]
    largest_n_indices = np.argsort(div_scores, axis=1)[:, -n:]
    return smallest_n_indices, largest_n_indices


def similarity(A, B, t="cosine"):
    if t == "cosine":
        A_normalized = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_normalized = B / np.linalg.norm(B, axis=1, keepdims=True)
        similarity = np.sum(A_normalized * B_normalized, axis=1)
    elif t == 'euc':
        similarity = np.linalg.norm(A - B, axis=1)
    return similarity


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
