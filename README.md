# Multi-View Differentiable Mesh Optimize

Optimises a COLMAP mesh texture and per-camera colour/exposure corrections.
Optional geometry offsets can be enabled with `--learn_geometry`. Camera poses
and intrinsics remain fixed. Training uses differentiable nvdiffrast rendering
and compares each render with its source photograph.

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

The scene should contain COLMAP files in `sparse/` and source images in
`images/`. The mesh can be supplied with `--mesh`, or placed as a `.ply` or
`.obj` file under `scene_root/mesh/`. `--uv_unwrap` requires `xatlas`.

## Optional live scene view

Append `--live_view` to the command to open an interactive OpenCV preview while
training. Use `--live_view_every N` to change the refresh interval.

## Outputs

The output directory contains the optimised texture, textured mesh, PPISP
parameters, loss data, loss curve, and resized-image cache. With
`--enable_skybox`, it also contains the optimised skybox texture and mesh.