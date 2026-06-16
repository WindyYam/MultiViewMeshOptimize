"""
Dataset loader for COLMAP + dense reconstruction output.

Expected directory layout:
  scene_root/
    sparse/
      cameras.txt        (or cameras.bin)
      images.txt         (or images.bin)
    images/
      <img_name>.jpg     (undistorted ground-truth photographs)
    mesh/
      textured_mesh.obj  (dense reconstruction with UV mapping)
      textured_mesh.mtl
      texture_*.png      (initial texture atlas)

This module reads camera poses, image paths, and the mesh into tensors
ready for the optimiser.
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _srgb_to_linear_np(rgb01: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0,1] to linear RGB."""
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return np.where(
        rgb01 <= 0.04045,
        rgb01 / 12.92,
        np.power((rgb01 + 0.055) / 1.055, 2.4),
    )


def _open_rgb_image(path: str) -> Image.Image:
    """Open an RGB image and allow trusted oversized textures when needed."""
    try:
        return Image.open(path).convert("RGB")
    except Image.DecompressionBombError:
        # Some reconstructed texture atlases legitimately exceed Pillow's
        # default safety threshold. Retry with the limit disabled for this load.
        old_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                return Image.open(path).convert("RGB")
        finally:
            Image.MAX_IMAGE_PIXELS = old_limit


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraView:
    cam_idx:   int
    image_path: str
    R:          torch.Tensor   # (3,3)
    t:          torch.Tensor   # (3,)
    K:          torch.Tensor   # (3,3)
    W:          int
    H:          int
    gt_image:   Optional[torch.Tensor] = None  # (H, W, 3) float32 [0,1]


@dataclass
class MeshData:
    vertices:  torch.Tensor   # (V, 3) world-space XYZ
    faces:     torch.Tensor   # (F, 3) int64
    uvs:       torch.Tensor   # (V, 2) UV coords in [0,1]
    face_uvs:  Optional[torch.Tensor] = None   # (F, 3, 2) per-face-vertex UV
    tex_image: Optional[torch.Tensor] = None   # (H_tex, W_tex, 3) float32 [0,1]


# ---------------------------------------------------------------------------
# COLMAP loader — pycolmap → binary .bin → text .txt  (in priority order)
# ---------------------------------------------------------------------------

# COLMAP camera model id → number of params before distortion coeffs
# Params layout: [f, cx, cy] or [fx, fy, cx, cy, ...]
_COLMAP_MODEL_PARAMS = {
    0:  ("SIMPLE_PINHOLE",    3),   # f, cx, cy
    1:  ("PINHOLE",           4),   # fx, fy, cx, cy
    2:  ("SIMPLE_RADIAL",     4),   # f, cx, cy, k
    3:  ("RADIAL",            5),   # f, cx, cy, k1, k2
    4:  ("OPENCV",            8),   # fx, fy, cx, cy, k1, k2, p1, p2
    5:  ("OPENCV_FISHEYE",    8),
    6:  ("FULL_OPENCV",      12),
    7:  ("FOV",               5),
    8:  ("SIMPLE_RADIAL_FISHEYE", 4),
    9:  ("RADIAL_FISHEYE",    5),
    10: ("THIN_PRISM_FISHEYE",12),
}
_SINGLE_F_MODELS = {0, 2, 3, 7, 8, 9}  # model ids where params[0] = single focal length


def _qvec_to_rotation(qvec) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def _params_to_intrinsics(model_id: int, params: list, W: int, H: int) -> dict:
    """Convert raw COLMAP params array to (fx, fy, cx, cy) regardless of model."""
    if model_id in _SINGLE_F_MODELS:
        fx = fy = float(params[0]); cx = float(params[1]); cy = float(params[2])
    elif len(params) >= 4:
        fx = float(params[0]); fy = float(params[1])
        cx = float(params[2]); cy = float(params[3])
    else:
        fx = fy = float(params[0]); cx = W / 2.0; cy = H / 2.0
    return dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)


# ---- pycolmap (handles bin + txt, multiple API versions) ----------------

def _load_colmap_pycolmap(sparse_dir: str) -> List[dict]:
    """Load via pycolmap — tolerates API differences across versions."""
    import pycolmap
    rec = pycolmap.Reconstruction(sparse_dir)

    def _maybe_call(x):
        return x() if callable(x) else x

    def _extract_pose(img):
        """
        Return (qvec_wxyz, tvec_xyz) for multiple pycolmap versions.
        Newer versions expose cam_from_world as property or method and do not
        expose qvec/tvec directly on Image.
        """
        # Newer pycolmap path
        if hasattr(img, "cam_from_world"):
            xform = _maybe_call(getattr(img, "cam_from_world"))
            if xform is not None:
                rot = _maybe_call(getattr(xform, "rotation", None))
                q = None
                if rot is not None and hasattr(rot, "quat"):
                    q = _maybe_call(getattr(rot, "quat"))
                elif hasattr(xform, "quat"):
                    q = _maybe_call(getattr(xform, "quat"))

                t = _maybe_call(getattr(xform, "translation", None))
                if t is None and hasattr(xform, "tvec"):
                    t = _maybe_call(getattr(xform, "tvec"))

                if q is not None and t is not None:
                    q = np.asarray(q, dtype=np.float64).reshape(-1)
                    t = np.asarray(t, dtype=np.float64).reshape(-1)
                    if q.size >= 4 and t.size >= 3:
                        # pycolmap Rotation3d.quat is commonly xyzw.
                        qvec = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
                        tvec = [float(t[0]), float(t[1]), float(t[2])]
                        return qvec, tvec

        # Older pycolmap path
        if hasattr(img, "qvec") and hasattr(img, "tvec"):
            qvec = list(map(float, img.qvec))
            tvec = list(map(float, img.tvec))
            return qvec, tvec

        raise AttributeError("Could not extract pose from pycolmap Image")

    cam_info = {}
    for cam_id, cam in rec.cameras.items():
        W, H = int(cam.width), int(cam.height)
        p    = list(cam.params)
        # model_id: newer pycolmap exposes cam.model_id (int),
        # older versions expose cam.model as an enum with .value
        try:
            mid = int(cam.model_id)
        except AttributeError:
            try:
                mid = int(cam.model.value)
            except Exception:
                mid = -1
        cam_info[cam_id] = _params_to_intrinsics(mid, p, W, H)

    records = []
    for img_id, img in rec.images.items():
        cam_id = img.camera_id
        qvec, tvec = _extract_pose(img)

        records.append(dict(
            img_id=img_id,
            qvec=qvec,
            tvec=tvec,
            cam_id=cam_id,
            name=img.name,
            cam_info=cam_info[cam_id],
        ))

    records.sort(key=lambda r: r["img_id"])
    return records


# ---- Pure-Python binary .bin reader (zero extra deps) -------------------

def _read_cameras_bin(path: str) -> Dict[int, dict]:
    """
    Read cameras.bin without pycolmap.
    Binary format: https://colmap.github.io/format.html#cameras-binary
    """
    import struct
    cameras = {}
    with open(path, "rb") as f:
        n_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_cams):
            cam_id   = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            W        = struct.unpack("<Q", f.read(8))[0]
            H        = struct.unpack("<Q", f.read(8))[0]
            n_params = _COLMAP_MODEL_PARAMS.get(model_id, ("UNKNOWN", 4))[1]
            params   = list(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[cam_id] = _params_to_intrinsics(model_id, params, W, H)
    return cameras


def _read_images_bin(path: str) -> List[dict]:
    """
    Read images.bin without pycolmap.
    Binary format: https://colmap.github.io/format.html#images-binary
    """
    import struct
    images = []
    with open(path, "rb") as f:
        n_imgs = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_imgs):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec   = list(struct.unpack("<4d", f.read(32)))   # w x y z
            tvec   = list(struct.unpack("<3d", f.read(24)))
            cam_id = struct.unpack("<I", f.read(4))[0]
            # Read null-terminated name
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            # Skip 2D point observations
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)   # each point2D: x(8) y(8) point3D_id(8)
            images.append(dict(img_id=img_id, qvec=qvec, tvec=tvec,
                               cam_id=cam_id, name=name))
    images.sort(key=lambda r: r["img_id"])
    return images


# ---- Text parsers (last resort) -----------------------------------------

def _parse_colmap_cameras_txt(path: str) -> Dict[int, dict]:
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts  = line.split()
            cam_id = int(parts[0])
            W, H   = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            # Map model name to id for consistent handling
            name_to_id = {v[0]: k for k, v in _COLMAP_MODEL_PARAMS.items()}
            mid = name_to_id.get(parts[1], 1)   # default PINHOLE
            cameras[cam_id] = _params_to_intrinsics(mid, params, W, H)
    return cameras


def _parse_colmap_images_txt(path: str) -> List[dict]:
    images = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts  = lines[i].split()
        img_id = int(parts[0])
        qvec   = list(map(float, parts[1:5]))
        tvec   = list(map(float, parts[5:8]))
        cam_id = int(parts[8])
        name   = parts[9]
        images.append(dict(img_id=img_id, qvec=qvec, tvec=tvec,
                           cam_id=cam_id, name=name))
        i += 2
    return images


# ---- Unified entry point ------------------------------------------------

def _load_colmap_sparse(sparse_dir: str) -> tuple:
    """
    Load COLMAP sparse from a directory.  Priority:
      1. pycolmap          — reads .bin and .txt, handles all camera models
      2. Binary .bin       — pure Python, no extra deps
      3. Text .txt         — plain text fallback

    Returns (records, cam_info_by_id).
    When cam_info is embedded per-record (pycolmap path), cam_info_by_id is None.
    """
    sparse_path = Path(sparse_dir)

    # 1) pycolmap
    try:
        import pycolmap
        records = _load_colmap_pycolmap(sparse_dir)
        print(f"[COLMAP] Loaded via pycolmap ({pycolmap.__version__}): "
              f"{len(records)} images")
        return records, None
    except ImportError:
        print("[COLMAP] pycolmap not installed — trying binary reader")
    except Exception as e:
        print(f"[COLMAP] pycolmap failed ({e}) — trying binary reader")

    # 2) Binary .bin
    cams_bin = sparse_path / "cameras.bin"
    imgs_bin = sparse_path / "images.bin"
    if cams_bin.exists() and imgs_bin.exists():
        try:
            cam_info = _read_cameras_bin(str(cams_bin))
            records  = _read_images_bin(str(imgs_bin))
            print(f"[COLMAP] Loaded via binary reader: {len(records)} images")
            return records, cam_info
        except Exception as e:
            print(f"[COLMAP] Binary reader failed ({e}) — trying text parser")

    # 3) Text .txt
    cams_txt = sparse_path / "cameras.txt"
    imgs_txt = sparse_path / "images.txt"
    if cams_txt.exists() and imgs_txt.exists():
        cam_info = _parse_colmap_cameras_txt(str(cams_txt))
        records  = _parse_colmap_images_txt(str(imgs_txt))
        print(f"[COLMAP] Loaded via text parser: {len(records)} images")
        return records, cam_info

    raise FileNotFoundError(
        f"No readable COLMAP files found in: {sparse_dir}\n"
        f"  Expected cameras.bin + images.bin  (or .txt equivalents)\n"
        f"  Tried: {cams_bin}, {imgs_bin}, {cams_txt}, {imgs_txt}"
    )


# ---------------------------------------------------------------------------
# PLY mesh loader
# ---------------------------------------------------------------------------

def load_ply(ply_path: str) -> MeshData:
    """
    Load a .ply file (binary_little_endian or ASCII) into MeshData.

    Handles the two UV layouts produced by dense reconstruction tools:

    Layout A — per-face-vertex UV list  (OpenMVS, Meshroom, most texturing pipelines)
        element face N
          property list uchar int  vertex_indices
          property list uchar float texcoord        ← 6 floats per tri: u0 v0 u1 v1 u2 v2
        Vertices have NO uv properties.
        We expand the mesh so every face-vertex gets its own unique vertex+UV entry.

    Layout B — per-vertex UV  (MeshLab exports, manual UV-unwrapped meshes)
        element vertex N
          property float texture_u / s / u / ...
          property float texture_v / t / v / ...
        UVs are stored directly on the vertex.
    """
    import struct

    with open(ply_path, "rb") as f:
        raw = f.read()

    # ------------------------------------------------------------------ header
    header_end = raw.find(b"end_header")
    if header_end == -1:
        raise ValueError("Not a valid PLY file (no end_header)")
    header     = raw[:header_end].decode("ascii", errors="ignore")
    # body starts after "end_header" + line ending (\n or \r\n)
    after_tag  = header_end + len("end_header")
    if raw[after_tag:after_tag+2] == b"\r\n":
        body_start = after_tag + 2
    elif raw[after_tag:after_tag+1] == b"\n":
        body_start = after_tag + 1
    else:
        body_start = after_tag + 1  # safe fallback

    lines     = [l.strip() for l in header.splitlines() if l.strip()]
    is_binary = any("binary_little_endian" in l for l in lines)

    # Parse texture filename from comment
    tex_file = None
    for l in lines:
        if l.startswith("comment TextureFile"):
            tex_file = l.split(None, 2)[-1].strip()

    PLY_FMT = {
        "char":"b","uchar":"B","short":"h","ushort":"H",
        "int":"i","uint":"I","float":"f","double":"d",
        "int8":"b","uint8":"B","int16":"h","uint16":"H",
        "int32":"i","uint32":"I","float32":"f","float64":"d",
    }

    # elements: ordered list of (name, count, props, list_props)
    #   props      = [(name, fmt_char)]           regular properties
    #   list_props = [(name, count_fmt, val_fmt)] list properties
    elements = []
    elem_map  = {}
    cur = None
    for l in lines:
        p = l.split()
        if not p:
            continue
        if p[0] == "element":
            cur = {"name": p[1], "count": int(p[2]), "props": [], "list_props": []}
            elements.append(cur)
            elem_map[p[1]] = cur
        elif p[0] == "property" and cur is not None:
            if p[1] == "list":
                # property list <count_type> <val_type> <name>
                cur["list_props"].append((p[4], PLY_FMT.get(p[2],"B"), PLY_FMT.get(p[3],"f")))
            else:
                cur["props"].append((p[2], PLY_FMT.get(p[1], "f")))

    # ------------------------------------------------------------------ readers
    def read_vertex_block(data, offset, elem):
        """Read all regular vertex properties into (N, P) float32 array."""
        fmt    = "<" + "".join(f for _, f in elem["props"])
        stride = struct.calcsize(fmt)
        n      = elem["count"]
        rows   = []
        for i in range(n):
            rows.append(struct.unpack_from(fmt, data, offset))
            offset += stride
        return np.array(rows, dtype=np.float32), offset

    def read_face_block(data, offset, elem):
        """
        Read face element which may have:
          - one list property for vertex indices
          - one list property for texcoords  (optional)
        Returns (faces, texcoords_per_face_vertex)
          faces:     list of [i0,i1,i2]
          texcoords: list of [u0,v0,u1,v1,u2,v2]  or None
        """
        n          = elem["count"]
        list_props = elem["list_props"]   # [(name, count_fmt, val_fmt), ...]

        # Identify which list prop is indices and which is texcoords
        idx_prop = None
        tex_prop = None
        for name, cfmt, vfmt in list_props:
            if name in ("vertex_indices", "vertex_index"):
                idx_prop = (cfmt, vfmt)
            elif name in ("texcoord", "texcoords", "texture_coordinates"):
                tex_prop = (cfmt, vfmt)

        if idx_prop is None and list_props:
            idx_prop = (list_props[0][1], list_props[0][2])

        faces     = []
        texcoords = [] if tex_prop else None

        for _ in range(n):
            # read vertex index list
            n_idx = struct.unpack_from("<" + idx_prop[0], data, offset)[0]
            offset += struct.calcsize("<" + idx_prop[0])
            idx_fmt = f"<{n_idx}" + idx_prop[1]
            idxs    = list(struct.unpack_from(idx_fmt, data, offset))
            offset += struct.calcsize(idx_fmt)

            # read texcoord list (if present)
            if tex_prop:
                n_tc   = struct.unpack_from("<" + tex_prop[0], data, offset)[0]
                offset += struct.calcsize("<" + tex_prop[0])
                tc_fmt  = f"<{n_tc}" + tex_prop[1]
                tc      = list(struct.unpack_from(tc_fmt, data, offset))
                offset += struct.calcsize(tc_fmt)
            else:
                tc = None

            # triangulate (support quads)
            tris = [(0,1,2)] if n_idx == 3 else [(0,1,2),(0,2,3)] if n_idx == 4 else []
            for a,b,c in tris:
                faces.append([idxs[a], idxs[b], idxs[c]])
                if texcoords is not None and tc and len(tc) >= (max(a,b,c)+1)*2:
                    texcoords.append([tc[a*2],tc[a*2+1],
                                      tc[b*2],tc[b*2+1],
                                      tc[c*2],tc[c*2+1]])

        return faces, texcoords

    # ------------------------------------------------------------------ read body
    vert_elem = elem_map.get("vertex")
    face_elem = elem_map.get("face")
    if vert_elem is None:
        raise ValueError("PLY has no vertex element")

    # Sanity-check: vertex stride × count should leave room for face block
    vert_stride = struct.calcsize("<" + "".join(f for _,f in vert_elem["props"]))
    expected_vert_bytes = vert_stride * vert_elem["count"]
    print(f"[load_ply]   body_start={body_start}  vert_stride={vert_stride}  "
          f"vert_bytes={expected_vert_bytes}  total_file={len(raw)}")

    offset = body_start
    vert_data, offset = read_vertex_block(raw, offset, vert_elem)
    print(f"[load_ply]   After vertex block: offset={offset}  "
          f"remaining={len(raw)-offset} bytes for faces")
    face_lists, face_texcoords = ([], None)
    if face_elem:
        face_lists, face_texcoords = read_face_block(raw, offset, face_elem)

    # ------------------------------------------------------------------ build tensors
    prop_names = [p[0] for p in vert_elem["props"]]
    print(f"[load_ply]   Vertex properties : {prop_names}")
    print(f"[load_ply]   Face list props   : {[lp[0] for lp in (face_elem or {}).get('list_props',[])]}")

    def _col(name):
        return prop_names.index(name) if name in prop_names else None

    def _col_first(*names):
        for n in names:
            i = _col(n)
            if i is not None:
                return i, n
        return None, None

    xi, yi, zi = _col("x"), _col("y"), _col("z")
    base_verts = torch.from_numpy(vert_data[:, [xi, yi, zi]])   # (V, 3)

    # Vertex colours (optional)
    ri, _, _ = _col_first("red","r"), _col_first("green","g"), _col_first("blue","b")
    ri2, _ = _col_first("red","r")
    gi2, _ = _col_first("green","g")
    bi2, _ = _col_first("blue","b")
    vert_colors = None
    if ri2 is not None and gi2 is not None and bi2 is not None:
        vc = vert_data[:, [ri2, gi2, bi2]]
        # Interpret mesh vertex colours as sRGB and convert to linear so
        # initialization matches the renderer's linear-light texture pipeline.
        vc01 = vc / 255.0 if vc.max() > 1.5 else np.clip(vc, 0.0, 1.0)
        vc_lin = _srgb_to_linear_np(vc01.astype(np.float32, copy=False))
        vert_colors = torch.from_numpy(vc_lin.astype(np.float32, copy=False))

    # ------------------------------------------------------------------ UV handling

    # --- Layout A: per-face-vertex texcoords list ---
    if face_texcoords and len(face_texcoords) == len(face_lists):
        print(f"[load_ply]   UV layout: per-face-vertex texcoord list  ({len(face_lists)} faces)")
        # Expand mesh: each face-vertex becomes a unique vertex
        new_verts  = []
        new_uvs    = []
        new_faces  = []
        for fi, (f, tc) in enumerate(zip(face_lists, face_texcoords)):
            tri = []
            for vi_local, orig_vi in enumerate(f):
                new_verts.append(base_verts[orig_vi])
                new_uvs.append([tc[vi_local*2], 1.0 - tc[vi_local*2+1]])  # flip V
                tri.append(len(new_verts) - 1)
            new_faces.append(tri)
        verts = torch.stack(new_verts)                          # (F*3, 3)
        uvs   = torch.tensor(new_uvs, dtype=torch.float32)     # (F*3, 2)
        faces = torch.tensor(new_faces, dtype=torch.int64)     # (F, 3)

    # --- Layout B: per-vertex UV properties ---
    else:
        ui, u_name = _col_first("texture_u","s","u","texcoord_u","texture_s","tex_u")
        vi, v_name = _col_first("texture_v","t","v","texcoord_v","texture_t","tex_v")
        if ui is not None and vi is not None:
            print(f"[load_ply]   UV layout: per-vertex  ('{u_name}' col {ui}, '{v_name}' col {vi})")
            # Keep internal UV convention consistent across OBJ/PLY layouts:
            # U in [0,1], V top-origin (image space). Most file formats store
            # per-vertex V with bottom-origin semantics, so we flip here.
            uvs = torch.from_numpy(vert_data[:, [ui, vi]]).clone()
            uvs[:, 1] = 1.0 - uvs[:, 1]
        else:
            print(f"[load_ply]   No UV found — generating spherical UVs")
            import math
            v = base_verts - base_verts.mean(0)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
            u_c = 0.5 + torch.atan2(v[:,0], v[:,2]) / (2*math.pi)
            v_c = 0.5 - torch.asin(v[:,1].clamp(-1,1)) / math.pi
            uvs = torch.stack([u_c, v_c], dim=1)
        verts = base_verts
        faces = torch.tensor(face_lists, dtype=torch.int64) if face_lists                 else torch.zeros((0,3), dtype=torch.int64)

    if not face_lists:
        print("[load_ply]   WARNING: No faces — PLY is a point cloud")

    # ------------------------------------------------------------------ texture image
    tex_image = None
    tex_dir   = os.path.dirname(ply_path)
    if tex_file:
        tex_path = os.path.join(tex_dir, tex_file)
        if os.path.exists(tex_path):
            img = _open_rgb_image(tex_path)
            tex_np = np.array(img, dtype=np.float32) / 255.0
            # Assume texture files are sRGB; convert to linear for internal use.
            tex_np = _srgb_to_linear_np(tex_np)
            tex_image = torch.from_numpy(tex_np.astype(np.float32, copy=False))
            print(f"[load_ply]   Loaded texture: {tex_file}  {tex_image.shape} (sRGB -> linear)")
        else:
            print(f"[load_ply]   WARNING: texture file not found: {tex_path}")
    elif vert_colors is not None and faces.shape[0] > 0:
        tex_image = _bake_vertex_colors_to_texture(verts, faces, uvs, vert_colors)

    print(f"[load_ply]   {verts.shape[0]} verts, {faces.shape[0]} faces")
    return MeshData(vertices=verts, faces=faces, uvs=uvs, tex_image=tex_image)


def _bake_vertex_colors_to_texture(
    verts: torch.Tensor,
    faces: torch.Tensor,
    uvs:   torch.Tensor,
    colors: torch.Tensor,
    res:   int = 1024,
) -> torch.Tensor:
    """
    Rasterise vertex colours into a UV texture atlas.
    Returns (res, res, 3) float32 in [0,1].
    Used to initialise the texture from a vertex-coloured PLY.
    """
    # Vectorized splat of all face-vertex colors into UV texel bins.
    uv_px = uvs.clone()
    uv_px[:, 0] = uv_px[:, 0] * (res - 1)
    # Internal UVs use image-space convention (V=0 at top), so map directly.
    uv_px[:, 1] = uv_px[:, 1] * (res - 1)

    face_vidx = faces.reshape(-1).long()
    px = uv_px[face_vidx, 0].round().clamp(0, res - 1).to(torch.int64)
    py = uv_px[face_vidx, 1].round().clamp(0, res - 1).to(torch.int64)
    lin = (py * res + px).cpu().numpy()

    col_np = colors[face_vidx].cpu().numpy().astype(np.float32, copy=False)
    tex_acc = np.zeros((res * res, 3), dtype=np.float32)
    np.add.at(tex_acc, lin, col_np)

    weight_flat = np.bincount(lin, minlength=res * res).astype(np.float32)
    tex_flat = tex_acc
    nz = weight_flat > 0
    tex_flat[nz] /= weight_flat[nz, None]

    tex = torch.from_numpy(tex_flat.reshape(res, res, 3))
    weight = torch.from_numpy(weight_flat.reshape(res, res))
    mask = weight > 0

    # Fill uncovered texels from nearest covered texel to avoid large black regions.
    if torch.any(mask):
        mask_np = mask.cpu().numpy().astype(bool, copy=False)
        tex_np = tex.cpu().numpy().astype(np.float32, copy=False)
        try:
            from scipy import ndimage
            _, (iy, ix) = ndimage.distance_transform_edt(~mask_np, return_indices=True)
            nearest = tex_np[iy, ix]
            base = torch.from_numpy(nearest.astype(np.float32, copy=False))
        except Exception:
            # Fallback if scipy is unavailable.
            from PIL import Image as PILImage
            t_np = (tex_np * 255).astype(np.uint8)
            small = PILImage.fromarray(t_np).resize((max(1, res // 4), max(1, res // 4)), PILImage.BILINEAR)
            filled = small.resize((res, res), PILImage.BILINEAR)
            base = torch.from_numpy(np.array(filled).astype(np.float32) / 255.0)
        tex = torch.where(mask.unsqueeze(-1), tex, base)
    return tex.clamp(0, 1)


# ---------------------------------------------------------------------------
# OBJ mesh loader (minimal, no external deps)
# ---------------------------------------------------------------------------

def load_obj(obj_path: str) -> MeshData:
    """
    Load a .obj file with UV coordinates.
    Returns a MeshData with per-vertex-position and per-face-vertex UVs.

    For meshes where face vertices share a position but differ in UV
    (very common for textured meshes), we duplicate vertices so that
    vertices[i] and uvs[i] are always in 1-1 correspondence.
    """
    verts_pos = []   # list of [x, y, z]
    verts_uv  = []   # list of [u, v]
    faces_pos = []   # list of [i0, i1, i2] into verts_pos
    faces_uv  = []   # list of [j0, j1, j2] into verts_uv
    tex_path  = None
    mtl_dir   = os.path.dirname(obj_path)

    with open(obj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] == "v":
                verts_pos.append([float(x) for x in tok[1:4]])
            elif tok[0] == "vt":
                verts_uv.append([float(tok[1]), float(tok[2])])
            elif tok[0] == "f":
                tri = tok[1:]
                if len(tri) == 4:
                    # Quad → 2 triangles
                    tri = [tri[0], tri[1], tri[2], tri[0], tri[2], tri[3]]
                    pairs = [(tri[0], tri[1], tri[2]), (tri[3], tri[4], tri[5])]
                else:
                    pairs = [tri]
                for t in pairs:
                    pi, ui = [], []
                    for token in t:
                        idxs = token.split("/")
                        pi.append(int(idxs[0]) - 1)
                        ui.append(int(idxs[1]) - 1 if len(idxs) > 1 and idxs[1] else 0)
                    faces_pos.append(pi)
                    faces_uv.append(ui)
            elif tok[0] == "mtllib":
                mtl_file = os.path.join(mtl_dir, tok[1])
                if os.path.exists(mtl_file):
                    with open(mtl_file) as mf:
                        for ml in mf:
                            ml = ml.strip()
                            if ml.lower().startswith("map_kd"):
                                tex_path = os.path.join(mtl_dir, ml.split()[-1])

    # Build unified vertex array (duplicate verts if needed for UV)
    # Each face-vertex gets its own unique vertex index
    new_verts = []
    new_uvs   = []
    new_faces = []

    for fi, (fp, fu) in enumerate(zip(faces_pos, faces_uv)):
        tri_idx = []
        for pi, ui in zip(fp, fu):
            new_verts.append(verts_pos[pi])
            new_uvs.append(verts_uv[ui] if verts_uv else [0.0, 0.0])
            tri_idx.append(len(new_verts) - 1)
        new_faces.append(tri_idx)

    verts_t = torch.tensor(new_verts, dtype=torch.float32)
    uvs_t   = torch.tensor(new_uvs,   dtype=torch.float32)
    uvs_t[:, 1] = 1.0 - uvs_t[:, 1]   # OBJ V=0 is bottom; flip to top-left origin
    faces_t = torch.tensor(new_faces, dtype=torch.int64)

    # Load initial texture
    tex_image = None
    if tex_path and os.path.exists(tex_path):
        img = _open_rgb_image(tex_path)
        tex_np = np.array(img, dtype=np.float32) / 255.0
        # Assume texture files are sRGB; convert to linear for internal use.
        tex_np = _srgb_to_linear_np(tex_np)
        tex_image = torch.from_numpy(tex_np.astype(np.float32, copy=False))

    return MeshData(
        vertices=verts_t,
        faces=faces_t,
        uvs=uvs_t,
        tex_image=tex_image,
    )


# ---------------------------------------------------------------------------
# Main scene dataset
# ---------------------------------------------------------------------------

def _find_and_load_mesh(scene_root, mesh_path=None):
    """Find and load a mesh. Supports .ply and .obj. mesh_path overrides auto-search."""
    from pathlib import Path as _Path
    if mesh_path is not None:
        p = _Path(mesh_path)
        print(f"[ColmapScene] Loading mesh: {p}")
        mesh = load_ply(str(p)) if p.suffix.lower() == ".ply" else load_obj(str(p))
        print(f"[ColmapScene]   {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
        return mesh
    mesh_dir = scene_root / "mesh"
    candidates = []
    if mesh_dir.exists():
        candidates = list(mesh_dir.glob("*.ply")) + list(mesh_dir.glob("*.obj"))
    if not candidates:
        candidates = list(scene_root.glob("*.ply")) + list(scene_root.glob("*.obj"))
    if not candidates:
        raise FileNotFoundError(
            f"No .ply or .obj mesh found under {scene_root}. "
            "Pass mesh_path='/path/to/your.ply' explicitly."
        )
    ply_files = [c for c in candidates if c.suffix.lower() == ".ply"]
    chosen = ply_files[0] if ply_files else candidates[0]
    print(f"[ColmapScene] Loading mesh: {chosen}")
    mesh = load_ply(str(chosen)) if chosen.suffix.lower() == ".ply" else load_obj(str(chosen))
    print(f"[ColmapScene]   {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
    return mesh


def _sample_texture_at_uvs(tex_image: torch.Tensor, uvs: torch.Tensor) -> torch.Tensor:
    """Sample texture atlas at per-vertex UVs. Returns (V,3) in [0,1]."""
    if tex_image is None:
        raise ValueError("Cannot sample vertex colors: mesh has no texture image")
    if tex_image.ndim != 3 or tex_image.shape[-1] != 3:
        raise ValueError("Texture image must have shape (H, W, 3)")
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        raise ValueError("UV tensor must have shape (V, 2)")

    tex = tex_image.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
    grid = uvs.float() * 2.0 - 1.0
    # Internal UVs are top-origin, matching grid_sample image coordinates.
    grid = grid.view(1, 1, -1, 2)  # (1,1,V,2)
    sampled = F.grid_sample(
        tex,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # (1,3,1,V)
    return sampled.squeeze(0).squeeze(1).T.clamp(0.0, 1.0).contiguous()  # (V,3)


def _unwrap_mesh_xatlas(mesh: MeshData, atlas_size: int) -> MeshData:
    """
    Re-unwrap mesh UVs with xatlas and initialize atlas from colors
    sampled on the previous texture.
    """
    if mesh.faces.shape[0] == 0:
        raise ValueError("Cannot unwrap a mesh without faces")
    if mesh.tex_image is None:
        raise ValueError("Cannot unwrap with texture transfer: input mesh has no texture")

    try:
        import xatlas
    except Exception as exc:
        raise RuntimeError(
            "xatlas is required for xatlas unwrap mode. Install it with: pip install xatlas"
        ) from exc

    atlas_size = int(max(32, atlas_size))
    print(f"[ColmapScene] Re-unwrapping input mesh with xatlas (size={atlas_size})")

    sampled_colors = _sample_texture_at_uvs(mesh.tex_image, mesh.uvs)

    verts_np = mesh.vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    faces_np = mesh.faces.detach().cpu().numpy().astype(np.int32, copy=False)
    colors_np = sampled_colors.detach().cpu().numpy().astype(np.float32, copy=False)

    # xatlas returns a vertex remap, reindexed face indices, and per-vertex UVs.
    vmapping, face_indices, uvs = xatlas.parametrize(verts_np, faces_np)

    vmapping = np.asarray(vmapping, dtype=np.int64)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)

    if face_indices.ndim != 2 or face_indices.shape[1] != 3:
        raise RuntimeError("xatlas unwrap output is inconsistent: face index shape is not (F,3)")
    if vmapping.shape[0] != uvs.shape[0]:
        raise RuntimeError("xatlas unwrap output is inconsistent: vmapping/uv size mismatch")

    new_verts_np = verts_np[vmapping]
    new_cols_np = colors_np[vmapping]
    new_uvs_np = uvs.copy()
    # Keep internal convention aligned with existing loaders.
    new_uvs_np[:, 1] = 1.0 - new_uvs_np[:, 1]

    new_verts_t = torch.from_numpy(new_verts_np.astype(np.float32, copy=False))
    new_uvs_t = torch.from_numpy(new_uvs_np.astype(np.float32, copy=False))
    new_cols_t = torch.from_numpy(new_cols_np.astype(np.float32, copy=False))
    new_faces_t = torch.from_numpy(face_indices.astype(np.int64, copy=False))

    tex_image = _bake_vertex_colors_to_texture(
        verts=new_verts_t,
        faces=new_faces_t,
        uvs=new_uvs_t,
        colors=new_cols_t,
        res=atlas_size,
    )

    print(
        f"[ColmapScene] xatlas unwrap done: {new_verts_t.shape[0]} verts, "
        f"{new_faces_t.shape[0]} faces, tex={tex_image.shape[1]}x{tex_image.shape[0]}"
    )
    return MeshData(
        vertices=new_verts_t,
        faces=new_faces_t,
        uvs=new_uvs_t,
        tex_image=tex_image,
    )



def load_image(path: str, W: int = None, H: int = None) -> torch.Tensor:
    """Load RGB image to float32 [0,1] tensor (H, W, 3)."""
    img = _open_rgb_image(path)
    if W is not None and H is not None:
        img = img.resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)


class ColmapScene:
    """
    Loads a COLMAP sparse reconstruction + dense mesh.

    Args:
        scene_root:    root directory (described above)
        image_scale:   downsample factor for GT images (1 = full res)
        max_cameras:   optional cap on number of cameras to load
        mesh_path:     explicit path to .ply or .obj mesh (auto-detected if omitted)
    """

    def __init__(
        self,
        scene_root:   str,
        image_scale:  float = 1.0,
        max_cameras:  Optional[int] = None,
        mesh_path:    Optional[str] = None,
        uv_unwrap_mode: str = "none",
        uv_unwrap_size: int = 0,
        device:       str = "cpu",
    ):
        self.root   = Path(scene_root)
        self.scale  = image_scale
        self.device = torch.device(device)

        print(f"[ColmapScene] Loading scene from: {self.root}")

        # --- Parse COLMAP  (pycolmap for .bin, text fallback) ---
        def _has_colmap_sparse_files(d: Path) -> bool:
            return (
                ((d / "cameras.bin").exists() and (d / "images.bin").exists())
                or ((d / "cameras.txt").exists() and (d / "images.txt").exists())
            )

        sparse_candidates = [self.root / "sparse", self.root / "sparse" / "0", self.root]
        sparse_dir = None
        for cand in sparse_candidates:
            if cand.exists() and _has_colmap_sparse_files(cand):
                sparse_dir = cand
                break

        if sparse_dir is None:
            raise FileNotFoundError(
                f"No COLMAP sparse files found under {self.root}. "
                "Expected cameras/images .bin or .txt in sparse/, sparse/0, or scene root."
            )

        if sparse_dir == self.root:
            print("[ColmapScene] sparse/ not found; using COLMAP files from scene root")

        colmap_imgs, colmap_cams = _load_colmap_sparse(str(sparse_dir))

        if max_cameras:
            colmap_imgs = colmap_imgs[:max_cameras]

        # --- Build CameraView list ---
        self.views: List[CameraView] = []
        skipped_missing_images = 0
        img_dir = self.root / "images"

        for idx, rec in enumerate(colmap_imgs):
            # cam_info is either embedded (pycolmap path) or looked up (text path)
            cam_info = rec.get("cam_info") or colmap_cams[rec["cam_id"]]
            W0, H0 = cam_info["W"], cam_info["H"]
            W = max(1, int(W0 * image_scale))
            H = max(1, int(H0 * image_scale))
            scale_x = W / W0
            scale_y = H / H0

            R = torch.from_numpy(_qvec_to_rotation(rec["qvec"]))
            t = torch.tensor(rec["tvec"], dtype=torch.float32)
            K = torch.tensor([
                [cam_info["fx"] * scale_x, 0,                        cam_info["cx"] * scale_x],
                [0,                        cam_info["fy"] * scale_y, cam_info["cy"] * scale_y],
                [0,                        0,                        1                        ],
            ], dtype=torch.float32)

            img_path = str(img_dir / rec["name"])
            if not os.path.exists(img_path):
                skipped_missing_images += 1
                continue

            view = CameraView(
                cam_idx=len(self.views),
                image_path=img_path,
                R=R, t=t, K=K, W=W, H=H,
            )
            self.views.append(view)

        print(f"[ColmapScene]   {len(self.views)} cameras loaded")
        if skipped_missing_images > 0:
            print(f"[ColmapScene]   Skipped {skipped_missing_images} cameras with missing image files")

        # --- Load mesh ---
        self.mesh = _find_and_load_mesh(self.root, mesh_path)
        unwrap_mode = str(uv_unwrap_mode or "none").lower()
        if unwrap_mode in ("xatlas", "open3d_xatlas"):
            unwrap_size = int(uv_unwrap_size)
            if unwrap_size <= 0:
                raise ValueError("uv_unwrap_size must be > 0 when uv_unwrap_mode is enabled")
            self.mesh = _unwrap_mesh_xatlas(self.mesh, atlas_size=unwrap_size)
        elif unwrap_mode != "none":
            raise ValueError(f"Unknown uv_unwrap_mode='{uv_unwrap_mode}'")

    def load_gt_images(self):
        """Load all ground-truth images into memory (lazy call)."""
        print("[ColmapScene] Loading ground-truth images...")
        for view in self.views:
            if os.path.exists(view.image_path):
                view.gt_image = load_image(view.image_path, view.W, view.H)
            else:
                print(f"  WARNING: image not found: {view.image_path}")
                view.gt_image = None

    def __len__(self):
        return len(self.views)

    def __getitem__(self, idx: int) -> CameraView:
        return self.views[idx]


# ---------------------------------------------------------------------------
# Synthetic test scene generator (for unit tests / demo without real data)
# ---------------------------------------------------------------------------

def make_synthetic_scene(
    num_cameras: int = 8,
    W: int = 256, H: int = 192,
    device: str = "cpu",
) -> Tuple["ColmapScene", MeshData]:
    """
    Generate a minimal synthetic scene: a flat textured quad viewed from
    several angles with random per-camera exposure offsets.

    Returns (views, mesh) compatible with the optimiser.
    """
    import math

    # Build a simple plane mesh  (2 triangles)
    verts = torch.tensor([
        [-1, -1, 0],
        [ 1, -1, 0],
        [ 1,  1, 0],
        [-1,  1, 0],
    ], dtype=torch.float32)
    faces = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int64)
    uvs_per_vert = torch.tensor([
        [0,0],[1,0],[1,1],[0,1]
    ], dtype=torch.float32)

    # Build face-vertex expanded mesh (same as load_obj output)
    new_v, new_uv, new_f = [], [], []
    for fi, f in enumerate(faces):
        tri = []
        for vi in f:
            new_v.append(verts[vi])
            new_uv.append(uvs_per_vert[vi])
            tri.append(len(new_v)-1)
        new_f.append(tri)

    mesh = MeshData(
        vertices=torch.stack(new_v),
        faces=torch.tensor(new_f, dtype=torch.int64),
        uvs=torch.stack(new_uv),
        tex_image=None,
    )

    # Camera poses on a circle around the plane
    K = torch.tensor([
        [W*0.8, 0, W/2],
        [0, W*0.8, H/2],
        [0, 0, 1],
    ], dtype=torch.float32)

    @dataclass
    class SyntheticScene:
        views: list
        mesh: MeshData
        def __len__(self): return len(self.views)
        def __getitem__(self, i): return self.views[i]

    views = []
    for i in range(num_cameras):
        angle = 2 * math.pi * i / num_cameras
        # Camera sits at (sin, 0, -3) looking at origin
        cx = math.sin(angle) * 0.3
        tz = -3.0
        t = torch.tensor([cx, 0.0, tz], dtype=torch.float32)
        R = torch.eye(3)

        # Synthetic ground truth: flat grey with random exposure
        exp = 0.5 + torch.rand(1).item()
        gt  = torch.clamp(torch.ones(H, W, 3) * 0.6 * exp, 0, 1)

        views.append(CameraView(
            cam_idx=i, image_path="", R=R, t=t, K=K, W=W, H=H, gt_image=gt
        ))

    return SyntheticScene(views=views, mesh=mesh)