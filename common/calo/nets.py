"""Neural network building blocks shared by the projects.

All velocity networks share the signature ``f(x, t, c)`` with
``x`` (B, D) flat samples, ``t`` (B, 1) in [0, 1] and ``c`` (B, C) the
(already standardised) condition.  Image-space nets reshape internally.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t * freqs * 20.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class FiLMBlock(nn.Module):
    """Pre-norm residual block with feature-wise conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_gb = nn.Linear(cond_dim, 2 * dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        g, b = self.to_gb(cond).chunk(2, dim=-1)
        h = h * (1.0 + g) + b
        return x + self.mlp(h)


# ---------------------------------------------------------------------------
# 1) flat residual MLP velocity
# ---------------------------------------------------------------------------
class ResMLP(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 192, depth: int = 4, cond_dim: int = 0):
        super().__init__()
        self.time = TimeEmbedding(hidden // 2)
        tdim = hidden // 2
        self.cdim = cond_dim
        cdim = tdim + cond_dim
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, cdim) for _ in range(depth)])
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, c=None):
        cond = self.time(t)
        if self.cdim > 0:
            cond = torch.cat([cond, c], dim=-1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(F.silu(h))


# ---------------------------------------------------------------------------
# 2) convolution + self-attention U-Net velocity
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * cout)

    def forward(self, x, cond):
        h = F.silu(self.norm(self.conv1(x)))
        g, b = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = h * (1.0 + g) + b
        return F.silu(self.conv2(h))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)
        a, _ = self.attn(t, t, t, need_weights=False)
        t = t + a
        t = t + self.ff(t)
        return t.transpose(1, 2).reshape(b, c, h, w)


class AttnUNet(nn.Module):
    """16x16 image U-Net with a 4x4 self-attention bottleneck."""

    def __init__(self, dim: int = 256, grid: int = 16, base: int = 32, cond_dim: int = 0, heads: int = 4):
        super().__init__()
        self.grid = grid
        self.time = TimeEmbedding(64)
        cdim = 64 + cond_dim
        self.cond_dim = cdim
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.d1 = ConvBlock(base, base, cdim)
        self.d2 = ConvBlock(base, base * 2, cdim)
        self.d3 = ConvBlock(base * 2, base * 2, cdim)
        self.attn = SelfAttention(base * 2, heads)
        self.mid = ConvBlock(base * 2, base * 2, cdim)

        self.u3 = ConvBlock(base * 4, base * 2, cdim)   # up(64) ++ 64
        self.u2 = ConvBlock(base * 3, base, cdim)       # up(64) ++ 32
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, c=None):
        b = x.shape[0]
        cond = self.time(t)
        if self.cond_dim > 64:
            cond = torch.cat([cond, c], dim=-1)
        h = x.view(b, 1, self.grid, self.grid)

        h0 = self.in_conv(h)
        h1 = self.d1(h0, cond)                      # 16
        h2 = self.d2(self.pool(h1), cond)           # 8
        h3 = self.d3(self.pool(h2), cond)           # 4
        h3 = self.attn(h3, cond)
        h3 = self.mid(h3, cond)

        u3 = self.u3(torch.cat([self.up(h3), h2], 1), cond)   # 8
        u2 = self.u2(torch.cat([self.up(u3), h1], 1), cond)   # 16
        return self.out(u2).flatten(1)


# ---------------------------------------------------------------------------
# 3) DeepSets velocity for point clouds
# ---------------------------------------------------------------------------
class DeepSets(nn.Module):
    """Permutation-equivariant point-cloud velocity.

    Input points ``(B, P, 4)`` = (x, y, layer, energy), mask ``(B, P)``.
    Output ``(B, P, 4)``.
    """

    def __init__(self, feat: int = 4, hidden: int = 96, cond_dim: int = 0, depth: int = 3):
        super().__init__()
        self.time = TimeEmbedding(48)
        cdim = 48 + cond_dim
        self.cond_dim = cdim

        layers = []
        cin = feat
        for _ in range(depth):
            layers += [nn.Linear(cin, hidden), nn.SiLU()]
            cin = hidden
        self.point_mlp = nn.Sequential(*layers)
        self.pool_proj = nn.Sequential(nn.Linear(2 * hidden + cdim, hidden), nn.SiLU())
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden + hidden + cdim + feat, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat),
        )
        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)

    def forward(self, x, t, c=None, mask=None):
        # x: (B, P, feat)
        cond = self.time(t)
        if self.cond_dim > 48:
            cond = torch.cat([cond, c], dim=-1)
        h = self.point_mlp(x)                                   # (B,P,H)
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device)
        m = mask[..., None]
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = (h * m - 1e9 * (1 - m)).max(1).values
        g = self.pool_proj(torch.cat([mean, mx, cond], dim=-1))  # (B,H)
        g = g[:, None, :].expand(-1, x.shape[1], -1)
        cc = cond[:, None, :].expand(-1, x.shape[1], -1)
        return self.out_mlp(torch.cat([h, g, cc, x], dim=-1))


# ---------------------------------------------------------------------------
# 4) Diffusion Transformer (DiT) with AdaLN-Zero
# ---------------------------------------------------------------------------
def _modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):
        sh, sc, g1, b1, g2, b2 = self.ada(F.silu(cond)).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), sh, sc) + b1[:, None, :]
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None, :] * a
        h = _modulate(self.norm2(x), sh, sc) + b2[:, None, :]
        x = x + g2[:, None, :] * self.mlp(h)
        return x


class DiT(nn.Module):
    def __init__(self, dim: int = 256, grid: int = 16, patch: int = 4,
                 hidden: int = 96, depth: int = 4, heads: int = 4, cond_dim: int = 0):
        super().__init__()
        self.grid = grid
        self.patch = patch
        self.hidden = hidden
        self.time = TimeEmbedding(64)
        self.cond_dim = 64 + cond_dim

        self.patch_embed = nn.Conv2d(1, hidden, kernel_size=patch, stride=patch)
        n_tok = (grid // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tok, hidden))
        nn.init.normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, self.cond_dim) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ada_out = nn.Linear(self.cond_dim, 2 * hidden)
        self.head = nn.Linear(hidden, patch * patch)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t, c=None):
        b = x.shape[0]
        cond = self.time(t)
        if self.cond_dim > 64:
            cond = torch.cat([cond, c], dim=-1)
        h = x.view(b, 1, self.grid, self.grid)
        tok = self.patch_embed(h).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            tok = blk(tok, cond)
        sh, sc = self.ada_out(F.silu(cond)).chunk(2, dim=-1)
        tok = _modulate(self.norm_out(tok), sh, sc)
        out = self.head(tok)                                    # (B, T, p*p)
        p = self.patch
        n = self.grid // p
        out = out.view(b, n, n, p, p).permute(0, 1, 3, 2, 4).reshape(b, self.grid, self.grid)
        return out.flatten(1)


# ---------------------------------------------------------------------------
# 5) RealNVP coupling flow (exact log-density)
# ---------------------------------------------------------------------------
class _Coupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 128):
        super().__init__()
        self.register_buffer("mask", mask)
        keep = int(mask.sum())
        change = dim - keep
        self.keep = keep
        self.net = nn.Sequential(
            nn.Linear(keep, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * change),
        )

    def forward(self, x, invert: bool = False):
        xa = x * self.mask
        xb = x * (1 - self.mask)
        s, t = self.net(xa).chunk(2, dim=-1)
        s = 0.7 * torch.tanh(s)          # keep exp(s) well behaved
        bmask = 1 - self.mask
        if not invert:
            yb = xb * torch.exp(s) + t
            logdet = (s * bmask).sum(-1)
        else:
            yb = (xb - t) * torch.exp(-s)
            logdet = -(s * bmask).sum(-1)
        return xa + yb, logdet


class RealNVP(nn.Module):
    def __init__(self, dim: int = 256, n_layers: int = 6, hidden: int = 160):
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_layers):
            m = torch.zeros(dim)
            m[i % 2::2] = 1.0
            masks.append(m)
        self.couplings = nn.ModuleList([_Coupling(dim, m, hidden) for m in masks])

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for c in self.couplings:
            z, ld = c(z, invert=False)
            logdet = logdet + ld
        base = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + logdet

    @torch.no_grad()
    def sample(self, n: int, device="cpu", temperature: float = 1.0) -> torch.Tensor:
        z = torch.randn(n, self.dim, device=device) * temperature
        for c in reversed(self.couplings):
            z, _ = c(z, invert=True)
        return z


# ---------------------------------------------------------------------------
# 6) small VAE for the latent-flow project
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, dim: int = 256, latent: int = 16, hidden: int = 256):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 2 * latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar.clamp(-8, 8)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar, z


# ---------------------------------------------------------------------------
# 7) small discriminator (benchmark project)
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 192, cond_dim: int = 0):
        super().__init__()
        self.time = TimeEmbedding(64) if cond_dim else None
        if cond_dim:
            self.cond = nn.Sequential(nn.Linear(cond_dim, 64), nn.SiLU())
        cin = dim + (64 if cond_dim else 0)
        self.net = nn.Sequential(
            nn.Linear(cin, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, c=None):
        if c is not None and self.cond is not None:
            x = torch.cat([x, self.cond(c)], dim=-1)
        return self.net(x)
