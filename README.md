# Texture + Per-Camera PPISP Joint Optimiser

Jointly optimises a mesh **texture map** and **per-camera ISP parameters** to resolve exposure/colour inconsistencies in COLMAP-reconstructed textured meshes.

Modelled after Gaussian Splatting training: random camera sampling, differentiable rendering, gradient-based optimisation.

---

## Problem

Dense reconstruction from multiple photos produces a textured mesh where the texture atlas is assembled from per-camera projections. Due to **exposure/white-balance inconsistencies** between photos, the seams are visible and large regions appear over- or under-exposed.

## Approach

```
World Mesh
    │
    ▼  vertex transform  (R, t, K  — fixed)
 Pixel-space UVs
    │
    ▼  triangle rasterisation + barycentric UV interpolation
 Raw HDR render  (H×W×3)   ← gradient flows back through here into texture
    │
    ▼  per-camera PPISP  (learnable)
        • Exposure multiplier      exp(log_e)
        • White balance RGB gains  exp(log_wb)
        • Vignette falloff         exp(-k·r²)
        • Gamma                    softplus(raw) + 0.5
        • Brightness / Contrast
    │
    ▼  LDR prediction  [0,1]
    │
    ▼  Photometric loss vs GT photo  (L1 + SSIM)
    │
    ▼  Backprop → update texture params + PPISP params
```

The mesh geometry and camera poses are **fixed** (COLMAP output). Only the texture values and ISP parameters are learnable.

---

## Directory structure

```
texture_optimizer/
├── __init__.py         – package exports
├── ppisp.py            – PPISPParams  (per-camera differentiable ISP model)
├── renderer.py         – TextureMap, Camera, SoftwareRasterizer, NvdiffrastRasterizer
├── losses.py           – PhotometricLoss, TextureRegLoss, PPISPRegLoss, TotalLoss
├── dataset.py          – ColmapScene, load_obj, make_synthetic_scene
└── trainer.py          – TexturePPISPTrainer, TrainConfig, train_scene

run_optimizer.py        – CLI entry point
```

---

## Quick start

### Synthetic test (no data required)
```bash
python run_optimizer.py --synthetic --iters 2000
```

### Real COLMAP scene
```bash
python run_optimizer.py \
    --scene /path/to/colmap_scene \
    --output outputs/ \
    --iters 10000 \
    --scale 0.5 \
    --tex_res 2048
```

### Python API
```python
from texture_optimizer import train_scene

trainer = train_scene(
    scene_root   = "/path/to/scene",
    output_dir   = "outputs",
    num_iterations = 10_000,
    image_scale  = 0.5,
    tex_res      = 2048,
)
```

---

## COLMAP scene layout expected

```
scene_root/
├── sparse/
│   ├── cameras.txt        (COLMAP intrinsics — PINHOLE or OPENCV model)
│   └── images.txt         (COLMAP extrinsics — quaternion + translation)
├── images/
│   └── *.jpg / *.png      (undistorted ground-truth photographs)
└── mesh/
    ├── textured_mesh.obj  (dense reconstruction with UV coordinates)
    ├── textured_mesh.mtl
    └── texture_*.png      (initial texture atlas — used as init)
```

For OpenMVS output, point `mesh/` at the folder containing `scene_mesh_texture.obj`.
For Meshroom output, use the `Texturing` node output folder.

---

## Key hyperparameters (`TrainConfig`)

| Parameter | Default | Notes |
|---|---|---|
| `num_iterations` | 5 000 | More iters → better texture, diminishing returns past ~15k |
| `warmup_iters` | 500 | PPISP-only phase — lets ISP calibrate before texture changes |
| `lr_texture` | 1e-3 | Reduce if texture oscillates |
| `lr_ppisp` | 5e-3 | PPISP converges fast; can be kept higher |
| `tex_reg_weight` | 5e-5 | TV smoothness on texture — increase if noisy |
| `ppisp_reg_weight` | 1e-2 | Keeps ISP near identity — increase to reduce per-camera over-fitting |
| `learn_vignette` | True | Disable with `--no_vignette` if vignette is pre-corrected |
| `image_scale` | 0.5 | Downsample GT images for speed — use 1.0 for final high-quality run |

---

## PPISP model parameters (per camera)

| Parameter | Space | Constraint | Description |
|---|---|---|---|
| `log_exposure` | log | ℝ | Global exposure multiplier `exp(x)` |
| `log_wb` | log | ℝ³ | Per-channel R/G/B white balance gains |
| `gamma_raw` | — | softplus + 0.5 | Display gamma (>0.5) |
| `brightness` | linear | ℝ | Additive brightness shift post-gamma |
| `log_contrast` | log | softplus + 0.5 | Multiplicative contrast (>0.5) |
| `log_vignette_k` | log | ℝ | Radial falloff strength `k` |

The ISP pipeline order is:  
`radiance × exposure → × WB → × vignette → gamma compress → contrast/brightness → clamp [0,1]`

---

## Rasteriser backends

The default `SoftwareRasterizer` is a pure-PyTorch triangle loop — correct and portable but **slow for large meshes**.

For production use, install [nvdiffrast](https://github.com/NVlabs/nvdiffrast):
```bash
pip install nvdiffrast
```
Then use `NvdiffrastRasterizer` in the trainer (swap in `renderer.py`).

For pytorch3d:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

---

## Outputs

```
outputs/
├── optimized_texture.png      – optimised texture atlas (PNG)
├── ppisp_params.json          – per-camera ISP params
├── loss_log.npy               – loss curve (numpy array)
├── checkpoint_*.pt            – training checkpoints
└── renders/
    └── cam_XXXX.png           – re-rendered views with optimised model
```

---

## Tips for real scenes

1. **Start with `image_scale=0.25`** for a fast exploratory run, then refine at 0.5–1.0.
2. **Check `ppisp_params.json`** after training — large exposure ratios (>3×) between cameras indicate the GT images were not consistently preprocessed.
3. **Increase `ppisp_reg_weight`** (e.g. 0.1) if you see per-camera over-fitting (renders look good individually but texture is patchy).
4. **Total variation reg** (`tex_reg_weight`) smooths the texture — lower it if you need sharp details, increase if noise is visible.
5. For very large meshes (>500k faces), the software rasterizer will be slow — use nvdiffrast or subsample cameras per iteration.
