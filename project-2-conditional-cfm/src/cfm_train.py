"""Continuous-time flow matching (CFM) train/sample helpers (torch, CPU)."""
import torch
import numpy as np


def cfm_train(model, Z, C, steps, batch=256, lr=1e-3, seed=0, val_frac=0.1, device="cpu"):
    rng = np.random.default_rng(seed)
    n = len(Z)
    n_val = max(1, int(n * val_frac))
    idx = rng.permutation(n)
    val, tr = idx[:n_val], idx[n_val:]
    Ztr = torch.as_tensor(Z[tr], dtype=torch.float32)
    Ctr = torch.as_tensor(C[tr], dtype=torch.float32)
    Zva = torch.as_tensor(Z[val], dtype=torch.float32)
    Cva = torch.as_tensor(C[val], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps):
        i = torch.randint(0, len(Ztr), (batch,), generator=g)
        x1 = Ztr[i]
        c = Ctr[i]
        t = torch.rand(x1.shape[0], 1, generator=g)
        x0 = torch.randn_like(x1)
        v_target = x1 - x0                       # straight CFM: x_t = (1-t) x0 + t x1 -> v = x1 - x0
        xt = (1 - t) * x0 + t * x1
        inp = torch.cat([xt, t], dim=-1)
        pred = model(inp, c)
        loss = F_loss()(pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        if (step + 1) % max(steps // 10, 1) == 0:
            with torch.no_grad():
                model.eval()
                j = torch.randperm(len(Zva))[:256]
                vt = Zva[j] - torch.randn_like(Zva[j])
                tt = torch.rand(len(j), 1)
                xt_v = torch.cat([(1 - tt) * torch.randn_like(Zva[j]) + tt * Zva[j], tt], -1)
                vl = float(F_loss()(model(xt_v, Cva[j]), vt))
                model.train()
            hist.append(("val", vl))
            print(f"  step {step+1:>5}/{steps}  train {hist[-2]:.4f}  val {vl:.4f}")
    return hist


def F_loss():
    import torch.nn as nn
    return nn.MSELoss()


@torch.no_grad()
def heun_sample(model, cond, n_odes=30, x0=None, device="cpu", seed=0):
    """Heun (2nd order) ODE integration of d x/dt = v(x, t)."""
    g = torch.Generator().manual_seed(seed)
    if x0 is None:
        x0 = torch.randn(len(cond), model.dim, generator=g)
    x = x0.clone()
    ts = torch.linspace(0.0, 1.0, n_odes + 1)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        tv = torch.full((len(x), 1), float(t0))
        v1 = model(torch.cat([x, tv], -1), cond)
        xh = x + (t1 - t0) * v1
        tv2 = torch.full((len(x), 1), float(t1))
        v2 = model(torch.cat([xh, tv2], -1), cond)
        x = x + 0.5 * (t1 - t0) * (v1 + v2)
    return x
