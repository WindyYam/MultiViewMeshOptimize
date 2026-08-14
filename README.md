# Multi-View Differentiable Mesh Optimize

Optimises a COLMAP mesh texture and per-camera colour/exposure corrections.
Optional geometry offsets can be enabled with `--learn_geometry`. Camera poses
and intrinsics remain fixed. Training uses differentiable nvdiffrast rendering
and compares each render with its source photograph.

## Example comparison

The video compares the coarse mesh, the geometry-optimized mesh, the
texture-optimized result, and the textured mesh exported directly
from COLMAP.

https://github.com/user-attachments/assets/7e48be0b-c135-4edf-8654-e6a66529d14c

## Dependencies

With the virtual environment activated, first install the PyTorch build with
CUDA support that matches your NVIDIA driver and CUDA platform. Do not install
the CPU-only PyTorch version. Use the official download page to select the
matching command:

<https://pytorch.org/get-started/locally/>

Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Start

Run from the `meshOptimize` directory and replace the placeholder paths:

```powershell
python .\run_optimizer.py `
    --scene C:\path\to\colmap_scene `
    --mesh C:\path\to\mesh.ply `
    --output C:\path\to\optimized `
    --iters 20000 `
    --tex_res 8192 `
    --uv_unwrap `
    --learn_geometry
```

The scene should contain source images in `scene_root/images/`. COLMAP files
are searched in `scene_root/sparse/`, `scene_root/sparse/0/`, or directly in
`scene_root/`. The mesh can be supplied with `--mesh` (resolved from the
current working directory), or placed as a `.ply` or `.obj` file under
`scene_root/mesh/` for automatic discovery. `--uv_unwrap` requires `xatlas`.
For best results, the coarse input mesh should include vertex colours or a
coarse texture atlas so texture optimization has a useful initialization.

## Optional live scene view

Append `--live_view` to the command to open an interactive OpenCV preview while
training. Use `--live_view_every N` to change the refresh interval.

## Outputs

The output directory contains the optimised texture, textured mesh, PPISP
parameters, loss data, loss curve, and resized-image cache. With
`--enable_skybox`, it also contains the optimised skybox texture and mesh.
