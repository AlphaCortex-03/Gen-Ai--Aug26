"""14 - Compressing the velocity net: int8 + pruning, size vs quality.

Post-training compression of the flow-matching velocity field:

* dynamic int8 quantization of every ``nn.Linear`` (PyTorch CPU path)
* global unstructured magnitude pruning at 50 / 70 / 90 %
* pruning + int8 combined

For each variant we record the serialized model size, sampling latency and the
C2ST / physics quality of the generated showers, then plot the Pareto front.
"""
from __future__ import annotations

import copy
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, split_data
from calo.nets import ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, save

import matplotlib.pyplot as plt


def size_kb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024.0


def nnz_fraction(model) -> float:
    nz = tot = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            nz += int((m.weight != 0).sum())
            tot += m.weight.numel()
    return nz / max(tot, 1)


def prune_model(model, sparsity: float):
    """Global unstructured magnitude pruning (post-training, no fine-tune)."""
    m = copy.deepcopy(model)
    weights = [p.abs().flatten() for n, p in m.named_parameters() if n.endswith("weight") and p.dim() > 1]
    if not weights:
        return m
    allw = torch.cat(weights)
    k = int(sparsity * len(allw))
    thresh = torch.sort(allw)[0][k] if k < len(allw) else allw.max() + 1
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("weight") and p.dim() > 1:
                p.mul_((p.abs() > thresh).float())
    return m


def quantize(model):
    return torch.quantization.quantize_dynamic(copy.deepcopy(model).eval(), {torch.nn.Linear}, dtype=torch.qint8)


def main() -> dict:
    args, cfg = setup(__file__, "p14-quantized-flow")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed)
    tr, te = split_data(data, 0.8, seed=seed)
    n_eval = P.n_eval(cfg, te)
    scaler, x1 = P.prep(cfg, tr)

    net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 192)), depth=int(cfg.get("depth", 4)), cond_dim=1)
    trainer = P.fit(cfg, net, scaler, tr, x1=x1)
    print(f"base model {count_params(net):,} params, {size_kb(net):.0f} kB")

    variants = {
        "fp32": net,
        "int8": quantize(net),
        "prune50": prune_model(net, 0.50),
        "prune70": prune_model(net, 0.70),
        "prune90": prune_model(net, 0.90),
        "prune90+int8": quantize(prune_model(net, 0.90)),
    }

    real = te["images"][:n_eval]
    e_eval = te["energy"][:n_eval]
    n_time = min(128, n_eval)
    rows, samples = [], {}
    for name, model in variants.items():
        torch.manual_seed(seed + 9)
        model.eval()
        t0 = time.perf_counter()
        _ = trainer.sample(n_time, e_eval[:n_time], steps=int(cfg["sample_steps"]), net=model)
        ms = 1000 * (time.perf_counter() - t0) / n_time

        gen = P.sample_images(trainer, e_eval, steps=int(cfg["sample_steps"]), seed=seed + 4, net=model)
        res = P.evaluate(real, gen, seed=seed)
        samples[name] = gen
        rows.append({
            "variant": name,
            "size_kb": size_kb(model),
            "nnz_frac": nnz_fraction(model) if not name.endswith("int8") or name == "fp32" else None,
            "ms_per_event": ms,
            "auc": res["c2st_auc"],
            "rel": res["mean_rel_err"],
            "depth": res["depth_gen"],
        })
        print(f"  {name:13s} {size_kb(model):7.1f} kB  {ms:6.2f} ms/ev  AUC {res['c2st_auc']:.3f}  rel {res['mean_rel_err']:.3f}")

    metrics = {
        "rows": rows,
        "table": fmt_table(rows, ["variant", "size_kb", "ms_per_event", "auc", "rel"]),
        "compression_x": rows[0]["size_kb"] / min(r["size_kb"] for r in rows),
        "quality_drop_auc": rows[0]["auc"] - min(r["auc"] for r in rows),
        "n_params": count_params(net),
    }

    # ---------------- figures -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for r in rows:
        axes[0].scatter(r["size_kb"], r["auc"], s=28)
        axes[0].annotate(r["variant"], (r["size_kb"], r["auc"]), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
        axes[1].scatter(r["ms_per_event"], r["auc"], s=28)
        axes[1].annotate(r["variant"], (r["ms_per_event"], r["auc"]), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    axes[0].set(xlabel="serialized size [kB]", ylabel="C2ST AUC", title="size vs quality")
    axes[1].set(xlabel="ms / event", ylabel="C2ST AUC", title="latency vs quality")
    axes[0].set_xscale("log")
    save(fig, fig_path(cfg, "pareto"))

    compare_grid(real, samples["fp32"], fig_path(cfg, "fp32_samples"), "fp32", n=4)
    compare_grid(real, samples["prune90"], fig_path(cfg, "prune90_samples"), "pruned 90%", n=4)

    np.savez_compressed(cfg["outdir"] / "samples.npz",
                        **{k.replace("+", "_"): v.astype(np.float32) for k, v in samples.items()})
    save_json(metric_path(cfg), metrics)
    print(metrics["table"])
    return metrics


if __name__ == "__main__":
    main()
