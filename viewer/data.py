"""
Loads an exported texture_optimizer bundle (optimized_mesh.ply,
optimized_texture.png, optimized_sh_coeff_XX.npy, sh_textures.json) from an
--output directory produced by run_optimizer.py / exporter.py.
"""

import json
import os

import numpy as np
from PIL import Image
from plyfile import PlyData


class ExportBundle:
    def __init__(self, vertices, faces, uvs, textures, sh_order):
        self.vertices = vertices      # (V,3) float32
        self.faces = faces            # (F,3) int32
        self.uvs = uvs                # (V,2) float32, PLY convention (v flipped vs. training)
        self.textures = textures      # list[np.ndarray], index 0 = base (0..1), 1..N = raw AC coeffs
        self.sh_order = sh_order


def _load_mesh_ply(path: str):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    vertices = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    uvs = np.stack([v["texture_u"], v["texture_v"]], axis=-1).astype(np.float32)

    face_idx = ply["face"].data["vertex_indices"]
    faces = np.stack([np.asarray(row, dtype=np.int32) for row in face_idx], axis=0)
    return vertices, faces, uvs


def _downsample(arr: np.ndarray, max_size: int) -> np.ndarray:
    h, w = arr.shape[0], arr.shape[1]
    long_side = max(h, w)
    if max_size is None or long_side <= max_size:
        return arr
    scale = max_size / float(long_side)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(new_h, new_w), mode="area")
    return t.squeeze(0).permute(1, 2, 0).numpy()


def load_export_bundle(output_dir: str, max_tex_size: int = None) -> ExportBundle:
    mesh_path = os.path.join(output_dir, "optimized_mesh.ply")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    vertices, faces, uvs = _load_mesh_ply(mesh_path)

    base_path = os.path.join(output_dir, "optimized_texture.png")
    base_img = np.asarray(Image.open(base_path).convert("RGB"), dtype=np.float32) / 255.0
    base_img = _downsample(base_img, max_tex_size)
    textures = [base_img]

    sh_order = 0
    sh_meta_path = os.path.join(output_dir, "sh_textures.json")
    if os.path.isfile(sh_meta_path):
        with open(sh_meta_path, "r", encoding="utf-8") as f:
            sh_meta = json.load(f)
        sh_order = int(sh_meta.get("sh_order", 0))
        coeffs = sorted(
            [c for c in sh_meta["coefficients"] if int(c["index"]) > 0],
            key=lambda c: int(c["index"]),
        )
        for c in coeffs:
            arr = np.load(os.path.join(output_dir, c["file"])).astype(np.float32)
            arr = _downsample(arr, max_tex_size)
            textures.append(arr)

    print(f"[Viewer] Mesh      : {vertices.shape[0]:,} verts, {faces.shape[0]:,} faces")
    print(f"[Viewer] SH order  : {sh_order}  ({len(textures)} texture layer(s))")
    for i, tex in enumerate(textures):
        print(f"[Viewer]   coeff {i}: {tex.shape[1]}x{tex.shape[0]}")

    return ExportBundle(vertices, faces, uvs, textures, sh_order)
