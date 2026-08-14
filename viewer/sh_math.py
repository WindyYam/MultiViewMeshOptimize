"""
Per-face tangent frame construction, mirroring
texture_optimizer.trainer.Trainer._compute_face_tangent_frames exactly.

The SH view-dependent shading in the trainer evaluates the real-SH basis in a
per-face (T, B, N) frame derived from the UV parameterization, not in world
space. To reproduce the trained appearance, the viewer must build the exact
same frame per face.
"""

import numpy as np

# Real SH basis constants, order 0..2 (must match trainer._eval_real_sh_basis).
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2_0 = 1.0925484305920792
SH_C2_1 = 1.0925484305920792
SH_C2_2 = 0.31539156525252005
SH_C2_3 = 1.0925484305920792
SH_C2_4 = 0.5462742152960396


def compute_face_tangent_frames(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray):
    """
    vertices: (V,3) float32
    faces:    (F,3) int
    uvs:      (V,2) float32, in the same convention used at training time
    Returns T, B, N each (F,3) float32.
    """
    verts = vertices.astype(np.float64)
    uv = uvs.astype(np.float64)
    f = faces.astype(np.int64)

    p0, p1, p2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    dp1 = p1 - p0
    dp2 = p2 - p0

    uv0, uv1, uv2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    n = np.cross(dp1, dp2)
    n = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8, None)

    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe = np.abs(det) > 1e-8
    inv_det = np.zeros_like(det)
    inv_det[safe] = 1.0 / det[safe]

    t = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) * inv_det[:, None]
    t = t - n * np.sum(t * n, axis=-1, keepdims=True)

    fallback_axis = np.zeros_like(n)
    fallback_axis[:, 0] = 1.0
    near_parallel = np.abs(n[:, 0]) > 0.9
    fallback_axis[near_parallel] = np.array([0.0, 1.0, 0.0])
    t_fallback = np.cross(fallback_axis, n)
    t_fallback = t_fallback / np.clip(np.linalg.norm(t_fallback, axis=-1, keepdims=True), 1e-8, None)

    t_norm = np.linalg.norm(t, axis=-1, keepdims=True)
    use_fallback = (t_norm <= 1e-8)[:, 0]
    t[use_fallback] = t_fallback[use_fallback]
    t[~use_fallback] = t[~use_fallback] / t_norm[~use_fallback]

    b = np.cross(n, t)
    handed = np.where(det >= 0.0, 1.0, -1.0)[:, None]
    b = b * handed
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)

    return t.astype(np.float32), b.astype(np.float32), n.astype(np.float32)
