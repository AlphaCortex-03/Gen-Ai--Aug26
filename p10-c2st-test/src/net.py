"""Tiny MLP velocity net shared by the torch Aug projects.

Convention (used by cfm_train.py & heun_sample.py):
  forward(x_t_and_t, c) -> v
  x already contains both the spatial state AND the time channel (dim = x_dim + 1);
  the conditioning scalar c is appended internally.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, x_dim, cond_dim=1, width=128):
        super().__init__()
        self.dim = x_dim                      # size of the sampled latent vector
        self.fc = nn.Sequential(
            nn.Linear(x_dim + 1 + cond_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, x_dim),
        )

    def forward(self, x, c):
        if c.dim() == 1:
            c = c[:, None]
        return self.fc(torch.cat([x, c], dim=-1))
