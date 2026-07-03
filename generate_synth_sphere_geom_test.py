#!/usr/bin/env python3
"""
Generate a synthetic COLMAP-style test scene for geometry-only optimization.

Scene contents:
- Ground-truth images: blue sphere (true radius) on white background.
- Optimization mesh: oversized blue sphere + white skybox cube.
- Sparse model: cameras.txt + images.txt (COLMAP text format).

Usage:
  python meshOptimize/scenes/generate_synth_sphere_geom_test.py
  python meshOptimize/scenes/generate_synth_sphere_geom_test.py --out meshOptimize/scenes/synth_sphere_geom
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def lookat_world_to_cam(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Return world->camera rotation R for x-right, y-down, z-forward convention.
    """
    z = normalize(target - cam_pos)          # forward
    x = normalize(np.cross(z, up))           # right
    y = normalize(np.cross(z, x))            # down for typical up=[0,1,0]
    return np.stack([x, y, z], axis=0).astype(np.float64)


def rotmat_to_qvec_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def render_blue_sphere(
    W: int,
    H: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    Rcw: np.ndarray,
    t: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Ray-sphere render in linear color, returns uint8 RGB image."""
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    x_cam = (xs - cx) / fx
    y_cam = (ys - cy) / fy
    z_cam = np.ones_like(x_cam)
    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    Rwc = Rcw.T
    dirs_world = dirs_cam @ Rwc.T
    cam_center = -Rwc @ t

    o = cam_center.reshape(1, 1, 3)
    d = dirs_world
    b = 2.0 * np.sum(o * d, axis=-1)
    c = np.sum(o * o, axis=-1) - radius * radius
    disc = b * b - 4.0 * c
    hit = disc >= 0.0

    img = np.ones((H, W, 3), dtype=np.float32)
    img[hit] = np.array([0.05, 0.2, 0.95], dtype=np.float32)
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def build_uv_sphere(
    radius: float,
    lat_steps: int,
    lon_steps: int,
    scale_xyz=(1.0, 1.0, 1.0),
):
    """Return OBJ-style vertex/uv/face lists for a UV sphere."""
    sx, sy, sz = [float(v) for v in scale_xyz]
    verts = []
    uvs = []
    faces = []

    for i in range(lat_steps + 1):
        v = i / lat_steps
        theta = math.pi * v
        y = math.cos(theta)
        r = math.sin(theta)
        for j in range(lon_steps + 1):
            u = j / lon_steps
            phi = 2.0 * math.pi * u
            x = r * math.cos(phi)
            z = r * math.sin(phi)
            verts.append([radius * sx * x, radius * sy * y, radius * sz * z])
            # Map sphere into left half (blue area) of texture atlas.
            u_mapped = 0.02 + 0.46 * u
            v_mapped = 0.02 + 0.96 * v
            uvs.append([u_mapped, v_mapped])

    row = lon_steps + 1
    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * row + j
            b = a + 1
            c = a + row
            d = c + 1
            faces.append([(a + 1, a + 1), (c + 1, c + 1), (b + 1, b + 1)])
            faces.append([(b + 1, b + 1), (c + 1, c + 1), (d + 1, d + 1)])
    return verts, uvs, faces


def build_skybox_cube(size: float, v_offset: int, vt_offset: int):
    """Return inward-facing skybox faces mapped to right half (white) of atlas."""
    s = size * 0.5
    corners = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
    ]
    verts = [list(c) for c in corners]

    uv_quad = [
        [0.55, 0.05],
        [0.95, 0.05],
        [0.95, 0.95],
        [0.55, 0.95],
    ]
    uvs = uv_quad * 6

    quads = [
        (0, 1, 2, 3),  # back
        (5, 4, 7, 6),  # front
        (4, 0, 3, 7),  # left
        (1, 5, 6, 2),  # right
        (3, 2, 6, 7),  # top
        (4, 5, 1, 0),  # bottom
    ]

    faces = []
    for qi, q in enumerate(quads):
        bvt = vt_offset + qi * 4
        a, b, c, d = q
        va = v_offset + a + 1
        vb = v_offset + b + 1
        vc = v_offset + c + 1
        vd = v_offset + d + 1
        # Inward orientation (camera is inside skybox).
        faces.append([(va, bvt + 1), (vb, bvt + 2), (vc, bvt + 3)])
        faces.append([(va, bvt + 1), (vc, bvt + 3), (vd, bvt + 4)])
    return verts, uvs, faces


def write_obj_with_mtl(
    scene_dir: Path,
    sphere_r_mesh: float,
    skybox_size: float,
    mesh_scale_xyz=(1.0, 1.0, 1.0),
):
    mesh_dir = scene_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    obj_path = mesh_dir / "synthetic_scene.obj"
    mtl_path = mesh_dir / "synthetic_scene.mtl"
    tex_path = mesh_dir / "atlas.png"

    sph_v, sph_vt, sph_f = build_uv_sphere(
        radius=sphere_r_mesh,
        lat_steps=24,
        lon_steps=48,
        scale_xyz=mesh_scale_xyz,
    )
    cube_v, cube_vt, cube_f = build_skybox_cube(
        size=skybox_size,
        v_offset=len(sph_v),
        vt_offset=len(sph_vt),
    )

    verts = sph_v + cube_v
    uvs = sph_vt + cube_vt
    faces = sph_f + cube_f

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl m0\n")
        f.write("Kd 1 1 1\n")
        f.write("Ka 0 0 0\n")
        f.write("Ks 0 0 0\n")
        f.write("d 1\n")
        f.write("illum 1\n")
        f.write("map_Kd atlas.png\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("mtllib synthetic_scene.mtl\n")
        f.write("usemtl m0\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for vt in uvs:
            f.write(f"vt {vt[0]:.8f} {vt[1]:.8f}\n")
        for tri in faces:
            a, b, c = tri
            f.write(f"f {a[0]}/{a[1]} {b[0]}/{b[1]} {c[0]}/{c[1]}\n")

    # Atlas: left half blue (sphere), right half white (skybox).
    tex = np.zeros((64, 128, 3), dtype=np.uint8)
    tex[:, :64, :] = np.array([13, 51, 242], dtype=np.uint8)
    tex[:, 64:, :] = np.array([255, 255, 255], dtype=np.uint8)
    Image.fromarray(tex, mode="RGB").save(tex_path)


def write_colmap_text(scene_dir: Path, records, W: int, H: int, fx: float, fy: float, cx: float, cy: float):
    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")

    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(records)}\n")
        for rec in records:
            q = rec["qvec"]
            t = rec["tvec"]
            f.write(
                f"{rec['id']} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} 1 {rec['name']}\n"
            )
            # Keep non-empty second line so the current parser's 2-line step stays aligned.
            f.write("0 0 -1\n")

    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# Empty points file for synthetic setup\n")


def build_dataset(
    out_dir: Path,
    num_cameras: int,
    W: int,
    H: int,
    radius_true: float,
    radius_mesh: float,
    cam_radius: float,
    mesh_scale_xyz=(1.0, 1.0, 1.0),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    fx = fy = 0.95 * float(W)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    records = []
    for i in range(num_cameras):
        ang = 2.0 * math.pi * (i / max(1, num_cameras))
        cam_pos = np.array([
            cam_radius * math.cos(ang),
            0.2 * math.sin(2.0 * ang),
            cam_radius * math.sin(ang),
        ], dtype=np.float64)

        Rcw = lookat_world_to_cam(cam_pos=cam_pos, target=target, up=up)
        t = -Rcw @ cam_pos
        qvec = rotmat_to_qvec_wxyz(Rcw)

        name = f"im_{i:03d}.png"
        img = render_blue_sphere(W, H, fx, fy, cx, cy, Rcw, t, radius=radius_true)
        Image.fromarray(img, mode="RGB").save(img_dir / name)

        records.append({
            "id": i + 1,
            "name": name,
            "qvec": qvec,
            "tvec": t,
        })

    write_colmap_text(out_dir, records, W, H, fx, fy, cx, cy)
    write_obj_with_mtl(
        out_dir,
        sphere_r_mesh=radius_mesh,
        skybox_size=20.0,
        mesh_scale_xyz=mesh_scale_xyz,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic sphere geometry test scene")
    p.add_argument("--out", type=str, default="meshOptimize/scenes/synth_sphere_geom")
    p.add_argument("--num_cameras", type=int, default=12)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--radius_true", type=float, default=0.8,
                   help="Sphere radius used to render GT images")
    p.add_argument("--radius_mesh", type=float, default=1.4,
                   help="Initial mesh sphere radius (intentionally too large)")
    p.add_argument("--mesh_flatten_axis", type=str, default="y", choices=["x", "y", "z"],
                   help="Axis to flatten for the initial mesh sphere")
    p.add_argument("--mesh_flatten_scale", type=float, default=1,
                   help="Scale factor on --mesh_flatten_axis for initial mesh sphere")
    p.add_argument("--cam_radius", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    mesh_scale = [1.0, 1.0, 1.0]
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    mesh_scale[axis_to_idx[str(args.mesh_flatten_axis).lower()]] = float(args.mesh_flatten_scale)
    build_dataset(
        out_dir=out_dir,
        num_cameras=max(1, int(args.num_cameras)),
        W=max(32, int(args.width)),
        H=max(32, int(args.height)),
        radius_true=float(args.radius_true),
        radius_mesh=float(args.radius_mesh),
        cam_radius=float(args.cam_radius),
        mesh_scale_xyz=tuple(mesh_scale),
    )
    print(f"[OK] Synthetic dataset generated at: {out_dir}")
    print(f"      Images: {out_dir / 'images'}")
    print(f"      Sparse: {out_dir / 'sparse'}")
    print(f"      Mesh:   {out_dir / 'mesh' / 'synthetic_scene.obj'}")


if __name__ == "__main__":
    main()
