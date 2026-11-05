"""08 - Flow distillation: 50-step teacher -> 1-step student (reflow).

The teacher is a standard conditional flow-matching model.  The student is
trained on *teacher ODE pairs* ``(x0, x1 = Teacher(x0))`` with the same linear
interpolation, which rectifies the trajectories so that a single Euler step is
enough.  We then sweep the number of function evaluations (NFE) for the teacher
and compare quality against wall-clock cost.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cfm import CFMTrainer, euler_sample
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, split_data
from calo.nets import ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, line_plot, loss_curve, save

import matplotlib.pyplot as plt

NFE_GRID = [1, 2, 4, 8, 16, 32, 50]


def main() -> dict:
    args, cfg = setup(__file__, "p8-flow-distillation: teacher -> 1-step student")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed)
    tr, te = split_data(data, 0.8, seed=seed)
    scaler, x1 = P.prep(cfg, tr)

    # ---------------- teacher -----------------------------------------
    teacher_net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 192)), depth=int(cfg.get("depth", 4)), cond_dim=1)
    teacher = P.fit(cfg, teacher_net, scaler, tr, x1=x1)
    print(f"teacher: {count_params(teacher_net):,} params, {teacher.train_seconds:.1f}s")

    # ---------------- teacher ODE pairs -------------------------------
    n_pairs = min(int(cfg.get("n_pairs", 4096)), len(tr["energy"]))
    torch.manual_seed(seed + 1)
    x0_pairs = torch.randn(n_pairs, 256)
    pair_energy = tr["energy"][:n_pairs]
    c_pairs = teacher.norm_cond(pair_energy)
    t0 = time.perf_counter()
    x1_pairs = euler_sample(teacher.net, x0_pairs, c_pairs, steps=int(cfg.get("teacher_steps", 50)))
    pair_time = time.perf_counter() - t0
    print(f"generated {n_pairs} teacher pairs in {pair_time:.1f}s")

    # ---------------- student (reflow) --------------------------------
    student_net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 192)), depth=int(cfg.get("depth", 4)), cond_dim=1)
    student = CFMTrainer(
        student_net, scaler,
        cond_mean=teacher.cond_mean, cond_std=teacher.cond_std,
        lr=float(cfg.get("student_lr", 8e-4)),
    )
    student.train(
        x1_pairs, pair_energy,
        steps=int(cfg.get("student_steps", 800)), batch=int(cfg["batch"]),
        x0=x0_pairs, log_every=200,
    )
    print(f"student: {count_params(student_net):,} params, {student.train_seconds:.1f}s")

    # ---------------- evaluation --------------------------------------
    n_eval = P.n_eval(cfg, te)
    real, e_eval = te["images"][:n_eval], te["energy"][:n_eval]
    c_eval = teacher.norm_cond(e_eval)
    torch.manual_seed(seed + 2)
    z0 = torch.randn(n_eval, 256)

    rows, teacher_samples = [], {}
    for nfe in NFE_GRID:
        t0 = time.perf_counter()
        z = euler_sample(teacher.net, z0, c_eval, steps=nfe)
        dt = time.perf_counter() - t0
        gen = teacher.decode(z)
        res = P.evaluate(real, gen, seed=seed)
        teacher_samples[nfe] = gen
        rows.append({
            "model": "teacher", "nfe": nfe, "auc": res["c2st_auc"],
            "rel": res["mean_rel_err"], "ms_per_event": 1000 * dt / n_eval,
        })
        print(f"  teacher NFE={nfe:2d}  AUC {res['c2st_auc']:.3f}  rel {res['mean_rel_err']:.3f}  {1000*dt/n_eval:.2f} ms/ev")

    student_rows = []
    for nfe in (1, 2, 4):
        t0 = time.perf_counter()
        z = euler_sample(student.ema, z0, c_eval, steps=nfe)
        dt = time.perf_counter() - t0
        gen = student.decode(z)
        res = P.evaluate(real, gen, seed=seed)
        student_rows.append({
            "model": "student", "nfe": nfe, "auc": res["c2st_auc"],
            "rel": res["mean_rel_err"], "ms_per_event": 1000 * dt / n_eval,
        })
        print(f"  student NFE={nfe}  AUC {res['c2st_auc']:.3f}  rel {res['mean_rel_err']:.3f}  {1000*dt/n_eval:.2f} ms/ev")

    rows += student_rows
    metrics = {
        "rows": rows,
        "table": fmt_table(rows, ["model", "nfe", "auc", "rel", "ms_per_event"]),
        "speedup_1step_vs_50": teacher_rows_speedup(rows),
        "n_params": count_params(teacher_net),
        "teacher_train_seconds": round(teacher.train_seconds, 2),
        "student_train_seconds": round(student.train_seconds, 2),
        "pair_generation_seconds": round(pair_time, 2),
    }

    # ---------------- figures -----------------------------------------
    loss_curve(teacher.history, fig_path(cfg, "teacher_loss"), "teacher MSE")
    loss_curve(student.history, fig_path(cfg, "student_loss"), "student reflow MSE")

    trows = [r for r in rows if r["model"] == "teacher"]
    line_plot([r["nfe"] for r in trows], {"teacher AUC": ([r["auc"] for r in trows], "o-")},
              fig_path(cfg, "quality_vs_nfe"), "NFE", "C2ST AUC", "teacher quality vs NFE",
              logx=True)
    line_plot([r["nfe"] for r in trows], {"latency": ([r["ms_per_event"] for r in trows], "o-")},
              fig_path(cfg, "latency_vs_nfe"), "NFE", "ms / event", "teacher latency vs NFE", logx=True)

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot([r["nfe"] for r in trows], [r["rel"] for r in trows], "o-", color="#2c7fb8", label="teacher")
    ax.plot([r["nfe"] for r in student_rows], [r["rel"] for r in student_rows], "s--", color="#d95f02", label="distilled student")
    ax.set_xscale("log")
    ax.set(xlabel="NFE", ylabel="mean physics rel. error", title="distillation: quality per NFE")
    ax.legend(frameon=False, fontsize=7)
    save(fig, fig_path(cfg, "rel_vs_nfe"))

    compare_grid(real, teacher_samples[50], fig_path(cfg, "teacher_50_vs_data"),
                 "teacher, 50 NFE", n=5)
    compare_grid(real, student.decode(euler_sample(student.ema, z0, c_eval, steps=1)),
                 fig_path(cfg, "student_1step_vs_data"), "student, 1 NFE", n=5)

    save_json(metric_path(cfg), metrics)
    print(metrics["table"])
    return metrics


def teacher_rows_speedup(rows):
    t50 = next(r for r in rows if r["model"] == "teacher" and r["nfe"] == 50)
    s1 = next(r for r in rows if r["model"] == "student" and r["nfe"] == 1)
    return {
        "latency_speedup": t50["ms_per_event"] / max(s1["ms_per_event"], 1e-9),
        "teacher_50_auc": t50["auc"], "student_1_auc": s1["auc"],
        "teacher_50_rel": t50["rel"], "student_1_rel": s1["rel"],
    }


if __name__ == "__main__":
    main()
