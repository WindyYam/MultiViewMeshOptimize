# Multi-View Differentiable Mesh Optimize

Optimises a COLMAP mesh texture and per-camera colour/exposure corrections.
PPISP models per-camera exposure differences so they do not bias the mesh and
texture optimization. Optional geometry offsets can be enabled with
`--learn_geometry`. Camera poses and intrinsics remain fixed. Training uses
differentiable nvdiffrast rendering and compares each render with its source
photograph.

## Example comparison

The video compares the coarse mesh, the geometry-optimized mesh, the
texture-optimized result, and the textured mesh exported directly
from COLMAP.

https://github.com/user-attachments/assets/7e48be0b-c135-4edf-8654-e6a66529d14c

## Comparison screenshots
Below are screenshots before and after the optimization on the scene "drjohnson"
<table>
    <tr>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141027.png" alt="Comparison screenshot 1"></td>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141057.png" alt="Comparison screenshot 2"></td>
    </tr>
    <tr>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141136.png" alt="Comparison screenshot 3"></td>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141148.png" alt="Comparison screenshot 4"></td>
    </tr>
    <tr>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141231.png" alt="Comparison screenshot 5"></td>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141241.png" alt="Comparison screenshot 6"></td>
    </tr>
    <tr>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141348.png" alt="Comparison screenshot 7"></td>
        <td><img src="assets/compare/Screenshot%202026-08-14%20141359.png" alt="Comparison screenshot 8"></td>
    </tr>
</table>

## Dependencies

Create and activate a virtual environment from the `meshOptimize` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use this activated environment for all subsequent commands in this README.

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

Optionally, if your COLMAP reconstruction uses distorted images, undistort
them before optimization so the images and camera model match:

```powershell
colmap image_undistorter `
    --image_path C:\path\to\colmap_scene\images `
    --input_path C:\path\to\colmap_scene\sparse\0 `
    --output_path C:\path\to\undistorted_scene `
    --output_type COLMAP
```

If you run this step, set `--scene` to the `--output_path` directory created
above. Otherwise, use the original COLMAP scene. Run from the `meshOptimize`
directory and replace the placeholder paths:

```powershell
python .\run_optimizer.py `
    --scene C:\path\to\colmap_scene `
    --mesh C:\path\to\mesh.ply `
    --output C:\path\to\optimized `
    --iters 10000 `
    --tex_res 8192 `
    --uv_unwrap `
    --learn_geometry
	--live_view
```

The scene should contain source images in `scene_root/images/`. COLMAP files
are searched in `scene_root/sparse/`, `scene_root/sparse/0/`, or directly in
`scene_root/`. The mesh can be supplied with `--mesh` (resolved from the
current working directory), or placed as a `.ply` or `.obj` file under
`scene_root/mesh/` for automatic discovery. `--uv_unwrap` requires `xatlas`.
For best results, the coarse input mesh should include vertex colours or a
coarse texture atlas so texture optimization has a useful initialization.

## Outputs

The output directory contains the optimised texture, textured mesh, PPISP
parameters, loss data, loss curve, and resized-image cache. With
`--enable_skybox`, it also contains the optimised skybox texture and mesh.
