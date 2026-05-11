"""
SILO (simplified single-file implementation)

Educational from-scratch implementation of "SILO: Solving Inverse Problems
with Latent Operators" (Raphaeli, Man, Elad - ICCV 2025, arXiv:2501.11746).

This is NOT a port of the official repo. It is a clean-room rewrite of the
paper's Algorithm 1 in one file, designed to be readable and hackable.

Algorithm reproduced:
    z_T ~ N(0, I)
    w   = clamp(E(y), -4, 4)
    for t = T..1:
        eps_hat   = UNet(z_t, t, prompt)
        z0_hat    = Tweedie(z_t, eps_hat)
        z'_{t-1}  = DDPMStep(z_t, eps_hat)
        w_hat     = H_theta(z0_hat, t)
        z_{t-1}   = z'_{t-1} - eta * grad_{z_t} ||w - w_hat||^2
    return D(z_0)

H_theta is the "latent degradation operator" - a small CNN trained to map
clean latents to the latents of their degraded counterparts. Once trained,
the autoencoder is used only at the very start (encode y) and very end
(decode z_0), which is the key efficiency / quality win in the paper.

USAGE (Colab):
    !pip install -q diffusers==0.30.0 transformers accelerate einops

    # Step A - train the operator (~30 min on T4 with 2000 steps)
    !python silo.py --mode train --task inpaint \
        --train_dir /content/ffhq_images --train_steps 2000

    # Step B - sample
    !python silo.py --mode sample --task inpaint \
        --test_image /content/ffhq_images/00001.png \
        --sample_steps 500 --eta 1.0

    # Or do both in one shot:
    !python silo.py --mode train_then_sample --task inpaint \
        --train_dir /content/ffhq_images

CAVEATS:
  * Won't match paper numbers - their operator is bigger and trained much
    longer. Expect "you can clearly see SILO working" quality, not SOTA.
  * Backprop through SD UNet at 512px is memory-heavy. T4 (16GB) just fits;
    if you OOM, drop --image_size to 384 or 256.
  * The "sr" task here uses bicubic-down-then-up (so latent shapes match).
    The paper's true SR uses smaller measurements and a shape-changing
    operator; that's a worthwhile extension for you to add.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# 1. Image-space degradations A(x). All take x in [-1, 1], (B,3,H,W).
# =============================================================================

def gaussian_kernel_1d(ksize: int, sigma: float, device, dtype):
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return g / g.sum()


def gaussian_blur(x, sigma=3.0):
    ksize = int(2 * round(3 * sigma) + 1)
    g = gaussian_kernel_1d(ksize, sigma, x.device, x.dtype)
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, ksize, ksize)
    return F.conv2d(x, kernel, padding=ksize // 2, groups=x.shape[1])


class Degradation:
    """Wraps an A(x) plus a measurement-noise level sigma_y."""

    def __init__(self, kind: str, sigma_y: float = 0.0, **kwargs):
        self.kind = kind
        self.sigma_y = sigma_y
        self.kwargs = kwargs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "inpaint":
            box = self.kwargs.get("box_size", 128)
            B, C, H, W = x.shape
            cy, cx = H // 2, W // 2
            y = x.clone()
            y[:, :, cy - box // 2:cy + box // 2, cx - box // 2:cx + box // 2] = 0.0

        elif self.kind == "gauss_blur":
            y = gaussian_blur(x, sigma=self.kwargs.get("sigma", 3.0))

        elif self.kind == "sr":
            # bicubic down-then-up so the spatial dims (and latent dims) stay
            # the same. This is a simplification - the paper does true SR with
            # a shape-changing operator. See module docstring.
            scale = self.kwargs.get("scale", 4)
            small = F.interpolate(x, scale_factor=1.0 / scale, mode="bicubic",
                                  align_corners=False, antialias=True)
            y = F.interpolate(small, scale_factor=float(scale), mode="bicubic",
                              align_corners=False)
        else:
            raise ValueError(f"Unknown degradation: {self.kind}")

        if self.sigma_y > 0:
            y = y + self.sigma_y * torch.randn_like(y)
        return y.clamp(-1, 1)


# =============================================================================
# 2. The latent degradation operator H_theta
#    Small time-conditioned residual CNN over latents.
#    Input  : (B, 4, h, w)  -- a "clean" latent z (or z0_hat at sampling)
#    Output : (B, 4, h, w)  -- emulated latent of A(decoded(z))
#    The output shape == input shape because all our A's preserve image dims.
# =============================================================================

class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[..., None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class LatentOperator(nn.Module):
    def __init__(self, in_ch: int = 4, hidden: int = 128,
                 t_dim: int = 128, n_blocks: int = 6):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbed(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(hidden, t_dim) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(8, hidden)
        self.out_conv = nn.Conv2d(hidden, in_ch, 3, padding=1)
        # zero-init the output conv so the network starts as identity (residual).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None].expand(z.shape[0])
        te = self.t_embed(t)
        h = self.in_conv(z)
        for blk in self.blocks:
            h = blk(h, te)
        delta = self.out_conv(F.silu(self.out_norm(h)))
        return z + delta  # residual: predicts the *change* needed to reach E(A(x))


# =============================================================================
# 2b. Our improvements over the vanilla CNN operator
#     1. Spatial Attention: learns which latent regions matter most
#     2. Multi-scale processing: captures detail at different kernel sizes
# =============================================================================

class SpatialAttention(nn.Module):
    """Learns a per-position importance map and uses it to gate features.

    Combines channel-wise mean and max statistics into a 2-channel summary,
    runs a 7x7 conv to mix spatial neighborhoods, then a sigmoid produces
    an attention map in (0, 1). Multiplied back onto the features.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        # mild init so attention starts ~uniform; the network learns to deviate
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # summarize along the channel axis: mean and max -> (B, 2, H, W)
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class MultiScaleBlock(nn.Module):
    """Three parallel branches with different kernel sizes, then fused.

    Branches: 3x3 (fine detail), 5x5 (mid), 7x7 (coarse / global structure).
    Outputs are concatenated along the channel axis and fused back to `ch`.
    Time conditioning is shared across branches.
    """

    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.t_proj = nn.Linear(t_dim, ch)
        per_branch = ch  # each branch keeps `ch` channels then we concat & fuse
        self.b3 = nn.Conv2d(ch, per_branch, 3, padding=1)
        self.b5 = nn.Conv2d(ch, per_branch, 5, padding=2)
        self.b7 = nn.Conv2d(ch, per_branch, 7, padding=3)
        self.fuse = nn.Conv2d(per_branch * 3, ch, 1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(x))
        h = h + self.t_proj(F.silu(t_emb))[..., None, None]
        cat = torch.cat([self.b3(h), self.b5(h), self.b7(h)], dim=1)
        h = self.fuse(cat)
        h = F.silu(self.norm2(h))
        h = self.attn(h)
        return x + h  # residual


class LatentOperatorPlus(nn.Module):
    """Improved H_theta: vanilla blocks + multi-scale blocks + spatial attention.

    Same input/output shape as LatentOperator, drop-in replacement everywhere.
    Same training objective. Same sampling algorithm. Just a smarter operator.
    """

    def __init__(self, in_ch: int = 4, hidden: int = 128,
                 t_dim: int = 128, n_res_blocks: int = 4, n_ms_blocks: int = 2):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbed(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, hidden, 3, padding=1)

        # interleave residual and multi-scale blocks: res, ms, res, ms, res, res
        # this gives the network both local refinement and multi-scale context
        blocks = []
        for i in range(n_res_blocks + n_ms_blocks):
            if i < n_ms_blocks * 2 and i % 2 == 1:
                blocks.append(MultiScaleBlock(hidden, t_dim))
            else:
                blocks.append(ResBlock(hidden, t_dim))
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.GroupNorm(8, hidden)
        self.out_conv = nn.Conv2d(hidden, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None].expand(z.shape[0])
        te = self.t_embed(t)
        h = self.in_conv(z)
        for blk in self.blocks:
            h = blk(h, te)
        delta = self.out_conv(F.silu(self.out_norm(h)))
        return z + delta


def make_operator(arch: str) -> nn.Module:
    """Factory for the operator. Lets the CLI pick which architecture to use."""
    if arch == "silo_v2":
        return LatentOperator()
    elif arch == "plus":
        return LatentOperatorPlus()
    else:
        raise ValueError(f"unknown arch: {arch}")


# =============================================================================
# 3. Dataset: a folder of clean images
# =============================================================================

class ImageFolderDataset(Dataset):
    EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, root: str, size: int = 512):
        self.paths = sorted(p for p in Path(root).rglob("*")
                            if p.suffix.lower() in self.EXTS)
        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # to [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


# =============================================================================
# 4. Stable Diffusion components: VAE, UNet, text encoder, scheduler
# =============================================================================

class SDComponents:
    def __init__(self, model_id: str, device: str, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        print(f"[SD] loading {model_id}")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=dtype).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        for net in (self.vae, self.unet, self.text_encoder):
            for p in net.parameters():
                p.requires_grad_(False)

        self.scaling = self.vae.config.scaling_factor  # 0.18215 for SD-v1.5

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x.to(self.dtype)).latent_dist.mean * self.scaling

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode((z / self.scaling).to(self.dtype)).sample.clamp(-1, 1)

    @torch.no_grad()
    def embed_prompt(self, prompt: str, negative: str = ""):
        def tok(p):
            return self.tokenizer(
                p, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(self.device)
        return self.text_encoder(tok(prompt))[0], self.text_encoder(tok(negative))[0]


# =============================================================================
# 5. Training H_theta
#    Loss: || H_theta(z + small_noise, t) - w ||_1
#    where z = E(x), w = E(A(x)). The small noise is so the operator is robust
#    to z0_hat (which is noisy at sampling time, especially at large t).
# =============================================================================

class EMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of the parameters that updates as
        ema_param = decay * ema_param + (1 - decay) * param
    after each optimizer step. EMA weights are typically smoother and
    generalize better than the raw final weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return {n: t.clone() for n, t in self.shadow.items()}


def cosine_lr_with_warmup(step: int, total: int, warmup: int,
                          base_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warmup for `warmup` steps, then cosine decay to `min_lr`."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_operator(sd: SDComponents, deg: Degradation, train_dir: str,
                   ckpt_path: str, steps: int = 2000, batch_size: int = 4,
                   lr: float = 1e-4, image_size: int = 512,
                   arch: str = "silo_v2",
                   val_frac: float = 0.05, val_every: int = 100,
                   ema_decay: float = 0.999, warmup_frac: float = 0.05,
                   log_path: str | None = None) -> nn.Module:

    ds_full = ImageFolderDataset(train_dir, size=image_size)
    if len(ds_full) == 0:
        raise RuntimeError(f"no images found under {train_dir}")

    # train/val split (deterministic for reproducibility)
    n_val = max(8, int(len(ds_full) * val_frac))
    n_train = len(ds_full) - n_val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        ds_full, [n_train, n_val], generator=g)
    print(f"[train] {len(ds_full)} images: {n_train} train / {n_val} val, "
          f"batch {batch_size}, {steps} steps, arch={arch}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, drop_last=False, pin_memory=True)

    op = make_operator(arch).to(sd.device)
    n_params = sum(p.numel() for p in op.parameters())
    print(f"[train] operator '{arch}' has {n_params:,} parameters")

    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    ema = EMA(op, decay=ema_decay)
    warmup_steps = int(steps * warmup_frac)

    T = sd.scheduler.config.num_train_timesteps
    log_rows = []  # (step, train_loss, val_loss, lr)

    def encode_pair(batch_x):
        with torch.no_grad():
            y_pix = deg(batch_x)
            z = sd.encode(batch_x)
            w = sd.encode(y_pix)
        t = torch.randint(0, T, (z.shape[0],), device=sd.device).long()
        noise_scale = 0.1 * (t.float() / T).view(-1, 1, 1, 1)
        z_in = z + noise_scale * torch.randn_like(z)
        return z_in, t, w

    @torch.no_grad()
    def validate():
        op.eval()
        total = 0.0
        n_batches = 0
        for vb in val_loader:
            vb = vb.to(sd.device)
            z_in, t, w = encode_pair(vb)
            pred = op(z_in, t)
            total += F.l1_loss(pred, w).item()
            n_batches += 1
        op.train()
        return total / max(1, n_batches)

    op.train()
    it = iter(train_loader)
    running = 0.0
    for step in range(1, steps + 1):
        # cosine LR with warmup
        cur_lr = cosine_lr_with_warmup(step - 1, steps, warmup_steps, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        try:
            x = next(it)
        except StopIteration:
            it = iter(train_loader)
            x = next(it)
        x = x.to(sd.device)

        z_in, t, w = encode_pair(x)
        pred = op(z_in, t)
        loss = F.l1_loss(pred, w)

        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(op)

        running = 0.95 * running + 0.05 * loss.item() if step > 1 else loss.item()

        if step % val_every == 0 or step == 1 or step == steps:
            val_loss = validate()
            log_rows.append({
                "step": step,
                "train_loss": running,
                "val_loss": val_loss,
                "lr": cur_lr,
            })
            print(f"[train] {step:5d}/{steps}  train(ema)={running:.4f}  "
                  f"val={val_loss:.4f}  lr={cur_lr:.2e}")

    # save EMA weights as the headline checkpoint; raw weights as backup
    torch.save({
        "state_dict": ema.state_dict(),         # EMA = the one used for inference
        "raw_state_dict": op.state_dict(),      # unaveraged weights, for reference
        "task": deg.kind,
        "kwargs": deg.kwargs,
        "arch": arch,
        "uses_ema": True,
        "training_log": log_rows,
    }, ckpt_path)
    print(f"[train] saved (with EMA weights) -> {ckpt_path}")

    # also write training log to CSV for plotting
    if log_path is None:
        log_path = ckpt_path.replace(".pt", "_trainlog.csv")
    with open(log_path, "w") as f:
        f.write("step,train_loss,val_loss,lr\n")
        for r in log_rows:
            f.write(f"{r['step']},{r['train_loss']:.6f},"
                    f"{r['val_loss']:.6f},{r['lr']:.6e}\n")
    print(f"[train] training log saved -> {log_path}")

    # load EMA weights into the returned model
    op.load_state_dict({**op.state_dict(), **ema.state_dict()}, strict=False)
    return op


# =============================================================================
# 6. Sampling: SILO Algorithm 1
# =============================================================================

def silo_sample(sd: SDComponents, op: LatentOperator,
                y: torch.Tensor, prompt: str,
                num_steps: int = 500, eta_scale: float = 1.0,
                guidance_scale: float = 1.0, clamp_w: float = 4.0,
                seed: int = 0) -> torch.Tensor:
    device = sd.device
    g = torch.Generator(device=device).manual_seed(seed)

    # encode the (degraded) measurement into the latent space
    with torch.no_grad():
        w = sd.encode(y).clamp(-clamp_w, clamp_w)

    pos_embed, neg_embed = sd.embed_prompt(prompt)

    sd.scheduler.set_timesteps(num_steps, device=device)
    timesteps = sd.scheduler.timesteps
    alphas_cumprod = sd.scheduler.alphas_cumprod.to(device)

    # init noisy latent
    z = torch.randn(w.shape, generator=g, device=device, dtype=w.dtype)

    op.eval()

    for i, t in enumerate(timesteps):
        # we need grad through z -> eps -> z0_hat -> H_theta(z0_hat) -> loss
        z = z.detach().requires_grad_(True)

        # ---- UNet forward (with optional CFG) ----
        if guidance_scale > 1.0:
            z_in = torch.cat([z, z], dim=0)
            embed = torch.cat([neg_embed, pos_embed], dim=0)
            eps_both = sd.unet(z_in, t, encoder_hidden_states=embed).sample
            eps_neg, eps_pos = eps_both.chunk(2)
            eps = eps_neg + guidance_scale * (eps_pos - eps_neg)
        else:
            eps = sd.unet(z, t, encoder_hidden_states=pos_embed).sample

        # ---- Tweedie: predicted clean latent z0_hat ----
        a_bar = alphas_cumprod[t]
        z0_hat = (z - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()

        # ---- Latent guidance: ||w - H_theta(z0_hat, t)||_2 (paper Alg. 1) ----
        t_b = t.repeat(z.shape[0]) if t.dim() == 0 else t
        # clamp z0_hat (can blow up at high t when sqrt(a_bar) is tiny)
        w_hat = op(z0_hat.clamp(-6, 6), t_b)
        diff = w - w_hat
        loss = (diff ** 2).sum().clamp(min=1e-8).sqrt()  # L2 norm, not squared
        grad = torch.autograd.grad(loss, z)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Standard DDPM step (no grad needed; eps and z are detached) ----
        with torch.no_grad():
            z_prev = sd.scheduler.step(eps.detach(), t, z.detach()).prev_sample

        # ---- SILO update ----
        z = (z_prev - eta_scale * grad).detach()

        if (i + 1) % 50 == 0 or i == 0 or i == len(timesteps) - 1:
            with torch.no_grad():
                resid = ((w - w_hat) ** 2).sum().sqrt().item()
            print(f"[sample] {i+1:4d}/{len(timesteps)}  t={int(t)}  "
                  f"||w - H(z0_hat)||={resid:.3f}")

    with torch.no_grad():
        x_hat = sd.decode(z)
    return x_hat


# =============================================================================
# 7. Image I/O helpers
# =============================================================================

def to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1) / 2
    arr = (x[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR for tensors in [-1, 1]. Higher is better. Same range = lower MSE."""
    a = (a.clamp(-1, 1) + 1) / 2
    b = (b.clamp(-1, 1) + 1) / 2
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-10:
        return 99.0
    return 10 * math.log10(1.0 / mse)


# Lazy global so we only build the LPIPS net once across all eval images.
_LPIPS_NET = None


def lpips_distance(a: torch.Tensor, b: torch.Tensor, device: str) -> float:
    """LPIPS perceptual distance. Lower is better. Tensors in [-1, 1]."""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        try:
            import lpips as lpips_lib
        except ImportError:
            print("[lpips] package not installed; run: pip install lpips")
            return float("nan")
        # AlexNet backbone matches the SILO paper's choice (Section 5.1)
        _LPIPS_NET = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in _LPIPS_NET.parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        d = _LPIPS_NET(a.to(device), b.to(device))
    return d.item()


def save_grid(clean: torch.Tensor, measurement: torch.Tensor,
              recon: torch.Tensor, out_path: str,
              labels=("Clean", "Measurement", "Reconstruction")):
    """Side-by-side grid: clean | measurement | recon, with labels above each."""
    pil_imgs = [to_pil(t) for t in (clean, measurement, recon)]
    w, h = pil_imgs[0].size
    label_h = 32
    grid = Image.new("RGB", (w * 3, h + label_h), (255, 255, 255))
    for i, im in enumerate(pil_imgs):
        grid.paste(im, (i * w, label_h))
    # try to draw labels; fall back silently if no font
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        for i, label in enumerate(labels):
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((i * w + (w - tw) // 2, 6), label, fill=(0, 0, 0), font=font)
    except Exception as e:
        print(f"[grid] label drawing skipped: {e}")
    grid.save(out_path)


def evaluate(sd, op, deg, test_dir: str, out_dir: str, prompt: str,
             num_images: int, sample_steps: int, eta: float,
             cfg: float, image_size: int, seed: int):
    """Run reconstruction on N test images, save grids, report PSNR table."""
    os.makedirs(out_dir, exist_ok=True)
    ds = ImageFolderDataset(test_dir, size=image_size)
    n = min(num_images, len(ds))
    print(f"[eval] running on {n} images from {test_dir}")
    print(f"[eval] task={deg.kind}  steps={sample_steps}  eta={eta}  cfg={cfg}")

    rows = []  # (filename, psnr_meas, psnr_recon, lpips_meas, lpips_recon)
    for i in range(n):
        x = ds[i][None].to(sd.device)
        fname = ds.paths[i].name
        print(f"\n[eval] [{i+1}/{n}] {fname}")

        with torch.no_grad():
            y = deg(x)

        x_hat = silo_sample(
            sd, op, y, prompt,
            num_steps=sample_steps, eta_scale=eta,
            guidance_scale=cfg, seed=seed,
        )

        psnr_meas   = psnr(y, x)
        psnr_recon  = psnr(x_hat, x)
        lpips_meas  = lpips_distance(y, x, sd.device)
        lpips_recon = lpips_distance(x_hat, x, sd.device)
        rows.append((fname, psnr_meas, psnr_recon, lpips_meas, lpips_recon))

        grid_path = os.path.join(out_dir, f"grid_{i:02d}_{Path(fname).stem}.png")
        save_grid(x, y, x_hat, grid_path)
        print(f"[eval]   PSNR  meas={psnr_meas:.2f}  recon={psnr_recon:.2f}  "
              f"(higher is better)")
        print(f"[eval]   LPIPS meas={lpips_meas:.4f}  recon={lpips_recon:.4f}  "
              f"(lower is better)")
        print(f"[eval]   -> {grid_path}")

    print("\n" + "=" * 86)
    print(f"{'file':28s}  {'PSNR meas':>10s}  {'PSNR recon':>10s}  "
          f"{'LPIPS meas':>11s}  {'LPIPS recon':>12s}")
    print("-" * 86)
    for fname, pm, pr, lm, lr in rows:
        print(f"{fname[:28]:28s}  {pm:10.2f}  {pr:10.2f}  "
              f"{lm:11.4f}  {lr:12.4f}")
    print("-" * 86)
    avg_pm  = sum(r[1] for r in rows) / len(rows)
    avg_pr  = sum(r[2] for r in rows) / len(rows)
    avg_lm  = sum(r[3] for r in rows) / len(rows)
    avg_lr  = sum(r[4] for r in rows) / len(rows)
    print(f"{'AVERAGE':28s}  {avg_pm:10.2f}  {avg_pr:10.2f}  "
          f"{avg_lm:11.4f}  {avg_lr:12.4f}")
    print("=" * 86)
    print(f"\nPSNR  change: {avg_pr - avg_pm:+.2f} dB    "
          f"(positive means recon closer to clean than measurement was)")
    print(f"LPIPS change: {avg_lr - avg_lm:+.4f}     "
          f"(NEGATIVE means recon perceptually closer to clean)")

    # write csv for easy import to slides
    csv_path = os.path.join(out_dir, "metrics_results.csv")
    with open(csv_path, "w") as f:
        f.write("file,psnr_measurement,psnr_reconstruction,"
                "lpips_measurement,lpips_reconstruction\n")
        for fname, pm, pr, lm, lr in rows:
            f.write(f"{fname},{pm:.4f},{pr:.4f},{lm:.4f},{lr:.4f}\n")
        f.write(f"AVERAGE,{avg_pm:.4f},{avg_pr:.4f},{avg_lm:.4f},{avg_lr:.4f}\n")
    print(f"[eval] results table saved to {csv_path}")


def load_image(path: str, size: int, device: str,
               dtype=torch.float32) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(img)[None].to(device, dtype)


# =============================================================================
# 8. CLI
# =============================================================================

TASK_DEFAULTS = {
    "inpaint":    dict(box_size=128),
    "gauss_blur": dict(sigma=3.0),
    "sr":         dict(scale=4),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["train", "sample", "train_then_sample", "eval"],
                   default="train_then_sample")
    p.add_argument("--task", choices=list(TASK_DEFAULTS), default="inpaint")
    p.add_argument("--train_dir", type=str, default="/content/ffhq_images")
    p.add_argument("--test_image", type=str, default=None,
                   help="image to degrade and recover; if None, uses first "
                        "image in --train_dir")
    p.add_argument("--ckpt", type=str, default="silo_op.pt")
    p.add_argument("--out_dir", type=str, default="silo_out")
    p.add_argument("--prompt", type=str,
                   default="a high quality photo of a face")
    p.add_argument("--model_id", type=str,
                   default="stablediffusionapi/realistic-vision-v51")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--sample_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eta", type=float, default=1.0,
                   help="guidance strength (paper's eta)")
    p.add_argument("--cfg", type=float, default=1.0,
                   help="classifier-free guidance scale (1.0 = off)")
    p.add_argument("--sigma_y", type=float, default=0.01,
                   help="measurement noise std")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["silo_v2", "plus"], default="silo_v2",
                   help="operator architecture: 'silo_v2' = paper-style CNN, "
                        "'plus' = our improvements (spatial attn + multi-scale)")
    p.add_argument("--test_dir", type=str, default=None,
                   help="(eval mode) folder of test images; defaults to "
                        "--train_dir if not given")
    p.add_argument("--num_eval", type=int, default=5,
                   help="(eval mode) how many test images to reconstruct")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[silo] device={device}")
    os.makedirs(args.out_dir, exist_ok=True)

    deg = Degradation(args.task, sigma_y=args.sigma_y, **TASK_DEFAULTS[args.task])
    sd = SDComponents(args.model_id, device)

    # ---- Eval (run on N images, save grids, print PSNR table) ----
    if args.mode == "eval":
        ckpt = torch.load(args.ckpt, map_location=device)
        ckpt_arch = ckpt.get("arch", "silo_v2")  # default for old checkpoints
        op = make_operator(ckpt_arch).to(device)
        op.load_state_dict(ckpt["state_dict"])
        print(f"[eval] loaded checkpoint with arch='{ckpt_arch}', "
              f"task='{ckpt.get('task')}'")
        if ckpt.get("task") != args.task:
            print(f"[WARN] checkpoint trained for task '{ckpt.get('task')}' "
                  f"but you're evaluating task '{args.task}'")
        test_dir = args.test_dir or args.train_dir
        evaluate(
            sd, op, deg, test_dir, args.out_dir, args.prompt,
            num_images=args.num_eval, sample_steps=args.sample_steps,
            eta=args.eta, cfg=args.cfg, image_size=args.image_size,
            seed=args.seed,
        )
        return

    # ---- Train ----
    if args.mode in ("train", "train_then_sample"):
        op = train_operator(
            sd, deg, args.train_dir, args.ckpt,
            steps=args.train_steps, batch_size=args.batch_size,
            lr=args.lr, image_size=args.image_size, arch=args.arch,
        )
    else:
        ckpt = torch.load(args.ckpt, map_location=device)
        ckpt_arch = ckpt.get("arch", "silo_v2")
        op = make_operator(ckpt_arch).to(device)
        op.load_state_dict(ckpt["state_dict"])
        print(f"[silo] loaded checkpoint with arch='{ckpt_arch}'")
        if ckpt.get("task") != args.task:
            print(f"[WARN] checkpoint trained for task '{ckpt.get('task')}' "
                  f"but you're sampling task '{args.task}'")

    # ---- Sample ----
    if args.mode in ("sample", "train_then_sample"):
        if args.test_image is not None:
            x = load_image(args.test_image, args.image_size, device)
        else:
            x = ImageFolderDataset(args.train_dir, args.image_size)[0][None].to(device)

        with torch.no_grad():
            y = deg(x)

        x_hat = silo_sample(
            sd, op, y, args.prompt,
            num_steps=args.sample_steps, eta_scale=args.eta,
            guidance_scale=args.cfg, seed=args.seed,
        )

        clean_path = os.path.join(args.out_dir, "clean.png")
        meas_path  = os.path.join(args.out_dir, "measurement.png")
        recon_path = os.path.join(args.out_dir, "reconstruction.png")
        grid_path  = os.path.join(args.out_dir, "grid.png")
        to_pil(x).save(clean_path)
        to_pil(y).save(meas_path)
        to_pil(x_hat).save(recon_path)
        save_grid(x, y, x_hat, grid_path)
        p_meas  = psnr(y, x)
        p_recon = psnr(x_hat, x)
        print(f"[silo] PSNR  measurement={p_meas:.2f}  recon={p_recon:.2f}  "
              f"gain={p_recon - p_meas:+.2f}")
        print(f"[silo] saved: {clean_path}, {meas_path}, {recon_path}, {grid_path}")


if __name__ == "__main__":
    main()
