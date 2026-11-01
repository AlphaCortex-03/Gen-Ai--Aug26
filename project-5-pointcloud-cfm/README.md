# Project 5 — Point-Cloud Flow Matching (DeepSets)

**What it does.** Treats a shower as a **variable-length point cloud** (x, y, z, E)
instead of a fixed image, and generates points directly with a permutation-invariant
velocity network. Voxelised clouds are evaluated against data with C2ST.

**Model.** DeepSets velocity field — only **40.8k parameters** (14× smaller than the
ResMLP image model).

**Result** (from `outputs/metrics.json`):

- voxelised C2ST AUC **0.980**
- mean observable relative error **3.1%** (vs 12.7% for the image ResMLP project 2)
- depth/W1 error 0.13, width/W1 error 0.049 — physical profiles closely matched
- sampling: 17 ms/event

**How to run:**

```
cd project-5-pointcloud-cfm
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (3D point clouds, marginals, profiles,
voxelised-vs-data) and `outputs/metrics.json`.
