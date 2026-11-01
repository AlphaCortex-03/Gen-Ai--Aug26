"""Conditional flow matching: trainer, Euler sampler, EMA helpers."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .nets import TimeEmbedding  # noqa: F401  (re-exported convenience)
from .util import count_params


class CFMTrainer:
    """Flow matching on the linear path ``x_t = (1-t) x0 + t x1``.

    The model learns ``v(x,t,c) = x1 - x0``; sampling integrates the ODE from
    ``t=0`` (noise) to ``t=1`` (data) with an explicit Euler scheme.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        scaler,
        cond_mean: float = 0.0,
        cond_std: float = 1.0,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cond_dropout: float = 0.0,
        ema_decay: float = 0.999,
    ):
        self.net = net.to(device)
        self.scaler = scaler
        self.device = device
        self.cond_mean = float(cond_mean)
        self.cond_std = float(max(cond_std, 1e-6))
        self.cond_dropout = float(cond_dropout)
        self.ema_decay = ema_decay
        self.lr = lr
        self.weight_decay = weight_decay

        self.ema = copy.deepcopy(net).to(device).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.dim = None
        self.history: list[dict] = []

    # -- helpers ---------------------------------------------------------
    def norm_cond(self, craw) -> torch.Tensor:
        c = torch.as_tensor(craw, dtype=torch.float32, device=self.device)
        if c.dim() == 1:
            c = c[:, None]
        return (c - self.cond_mean) / self.cond_std

    @torch.no_grad()
    def _ema_update(self, decay: float) -> None:
        for pe, p in zip(self.ema.parameters(), self.net.parameters()):
            pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for be, b in zip(self.ema.buffers(), self.net.buffers()):
            be.copy_(b)

    @property
    def n_params(self) -> int:
        return count_params(self.net)

    # -- training --------------------------------------------------------
    def train(
        self,
        x1,
        craw,
        steps: int = 400,
        batch: int = 128,
        log_every: int = 100,
        verbose: bool = True,
        warmup: int = 20,
        t_max: float = 1.0,
        sigma_min: float = 0.0,
        x0: torch.Tensor | None = None,
    ) -> list[dict]:
        x1 = torch.as_tensor(x1, dtype=torch.float32, device=self.device)
        c = self.norm_cond(craw)
        n = x1.shape[0]
        if x0 is not None:
            # fixed noise pairing (used by reflow distillation)
            x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        self.dim = x1.shape[1]
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=max(steps, 1), pct_start=0.15
        )
        self.net.train()
        self.history = []
        t0 = time.perf_counter()
        # With very short runs a 0.999 EMA never catches up, so shorten its
        # horizon to ~1/10 of the run.
        ema_decay = min(self.ema_decay, max(0.0, 1.0 - 10.0 / max(steps, 10)))

        for step in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), device=self.device)
            xb = x1[idx]
            cb = c[idx]
            if self.cond_dropout > 0:
                keep = (torch.rand(cb.shape[0], 1, device=self.device) > self.cond_dropout).float()
                cb = cb * keep
            # optional noise floor: matches low-temperature sampling / data aug
            xb = xb + sigma_min * torch.randn_like(xb) if sigma_min > 0 else xb

            t = torch.rand(xb.shape[0], 1, device=self.device) * t_max
            if x0 is None:
                x0b = torch.randn_like(xb)
            else:
                x0b = x0[idx]
            xt = (1.0 - t) * x0b + t * xb
            target = xb - x0b
            pred = self.net(xt, t, cb)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step < warmup:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            self._ema_update(ema_decay)

            if step % log_every == 0 or step == steps - 1:
                rec = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "lr": float(sched.get_last_lr()[0]),
                    "elapsed_s": time.perf_counter() - t0,
                }
                self.history.append(rec)
                if verbose:
                    print(f"  step {step:5d}  loss {rec['loss']:.5f}  {rec['elapsed_s']:6.1f}s")

        self.train_seconds = time.perf_counter() - t0
        self.net.eval()
        return self.history

    # -- sampling --------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n: int,
        craw,
        steps: int = 50,
        guidance: float | None = None,
        net: torch.nn.Module | None = None,
        trajectory: bool = False,
        x_init: torch.Tensor | None = None,
    ):
        net = net if net is not None else self.ema
        was_training = net.training
        net.eval()
        dim = self.dim or 256
        x = torch.randn(n, dim, device=self.device) if x_init is None else x_init.clone()
        c = self.norm_cond(craw)
        traj = [x.clone()]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        for i in range(steps):
            t = ts[i].expand(n, 1)
            v = net(x, t, c)
            if guidance is not None:
                v_u = net(x, t, torch.zeros_like(c))
                v = v_u + guidance * (v - v_u)
            x = x + (ts[i + 1] - ts[i]) * v
            if trajectory:
                traj.append(x.clone())
        if was_training:
            net.train()
        return (x, traj) if trajectory else x

    # -- decoding --------------------------------------------------------
    def decode(self, z: torch.Tensor) -> np.ndarray:
        """Scaled sample -> physical images summing to 1, shape (N, G, G)."""
        img = self.scaler.inverse(z.detach().cpu().numpy())
        img = np.clip(img, 0.0, None)
        s = img.reshape(len(img), -1).sum(1, keepdims=True)
        img = img / np.maximum(s, 1e-12)
        g = int(round(np.sqrt(img.shape[1])))
        return img.reshape(len(img), g, g)


@torch.no_grad()
def euler_sample(
    net: torch.nn.Module,
    x0: torch.Tensor,
    cond: torch.Tensor,
    steps: int,
    guidance: float | None = None,
    return_traj: bool = False,
    clamp: float | None = None,
):
    """Integrate ``dx/dt = v(x,t,c)`` for ``steps`` Euler steps from ``x0``."""
    net.eval()
    n = x0.shape[0]
    x = x0.clone()
    traj = [x.clone()]
    ts = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    for i in range(steps):
        t = ts[i].expand(n, 1)
        v = net(x, t, cond)
        if guidance is not None:
            v_u = net(x, t, torch.zeros_like(cond))
            v = v_u + guidance * (v - v_u)
        x = x + (ts[i + 1] - ts[i]) * v
        if clamp is not None:
            x = x.clamp(-clamp, clamp)
        if return_traj:
            traj.append(x.clone())
    return (x, traj) if return_traj else x


@torch.no_grad()
def make_pairs(net, x0: torch.Tensor, cond: torch.Tensor, steps: int):
    """Teacher ODE pairs (noise, endpoint) for reflow distillation."""
    return x0, euler_sample(net, x0, cond, steps)


def ema_from(net: torch.nn.Module, decay: float = 0.999) -> torch.nn.Module:
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: torch.nn.Module, net: torch.nn.Module, decay: float) -> None:
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
