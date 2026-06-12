"""
Loss functions for texture + per-camera PPISP optimization.

Losses:
  - PhotometricLoss   : L1 + perceptual-style SSIM in LDR space
  - PPISPRegLoss      : soft regularisation to keep ISP params near identity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .ppisp import PPISPParams
    from .renderer import TextureMap

try:
    from pytorch_msssim import ssim as msssim_ssim
except Exception:
    msssim_ssim = None


# ---------------------------------------------------------------------------
# SSIM (structural similarity) — differentiable, no external deps
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


_SSIM_WINDOW: Optional[torch.Tensor] = None
_SSIM_WIN_SIZE = 11


def _ssim_loss_native(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    SSIM-based loss in [0, 1].  Lower is better.
    pred / target: (H, W, 3) float32 in [0, data_range]
    """
    global _SSIM_WINDOW
    device = pred.device

    if _SSIM_WINDOW is None or _SSIM_WINDOW.device != device:
        _SSIM_WINDOW = _gaussian_kernel(_SSIM_WIN_SIZE).to(device)

    kernel = _SSIM_WINDOW.to(dtype=pred.dtype).view(1, 1, _SSIM_WIN_SIZE, _SSIM_WIN_SIZE).expand(3, 1, -1, -1)
    p = pred.permute(2, 0, 1).unsqueeze(0)      # (1, 3, H, W)
    t = target.permute(2, 0, 1).unsqueeze(0)    # (1, 3, H, W)
    pad = _SSIM_WIN_SIZE // 2

    mu1 = F.conv2d(p, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(t, kernel, padding=pad, groups=3)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(p * p, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(t * t, kernel, padding=pad, groups=3) - mu2_sq
    sigma12   = F.conv2d(p * t, kernel, padding=pad, groups=3) - mu12

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / torch.clamp(denom, min=1e-8)

    if mask is None:
        return 1.0 - ssim_map.mean()

    valid = mask.to(device=pred.device, dtype=pred.dtype)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif valid.ndim == 3:
        valid = valid.unsqueeze(0)               # (1,1,H,W) or (1,3,H,W)

    if valid.shape[1] == 1:
        valid = valid.expand(-1, ssim_map.shape[1], -1, -1)

    denom = valid.sum().clamp(min=1e-8)
    ssim_mean = (ssim_map * valid).sum() / denom
    return 1.0 - ssim_mean


def _ssim_loss_msssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fast SSIM path using pytorch_msssim if available.
    pred / target: (H, W, 3) float32 in [0, data_range]
    """
    if msssim_ssim is None or mask is not None:
        return _ssim_loss_native(pred, target, data_range=data_range, mask=mask)

    p = pred.permute(2, 0, 1).unsqueeze(0)
    t = target.permute(2, 0, 1).unsqueeze(0)
    try:
        ssim_val = msssim_ssim(
            p,
            t,
            data_range=data_range,
            size_average=True,
            nonnegative_ssim=True,
        )
    except TypeError:
        ssim_val = msssim_ssim(
            p,
            t,
            data_range=data_range,
            size_average=True,
        )
    return 1.0 - ssim_val


def compute_face_normals_and_double_area(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n_raw = torch.cross(v1 - v0, v2 - v0, dim=1)
    a2 = n_raw.norm(dim=1).clamp(min=1e-12)
    n_unit = n_raw / a2.unsqueeze(1)
    return n_unit, a2


def _geometry_normal_tv_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_adj_pairs: Optional[torch.Tensor],
    face_adj_edges: Optional[torch.Tensor] = None,
    base_face_pair_weight: Optional[torch.Tensor] = None,
    weld_index: Optional[torch.Tensor] = None,
    num_weld_groups: Optional[int] = None,
    weld_group_inv_counts: Optional[torch.Tensor] = None,
    delta: float = 1e-3,
) -> torch.Tensor:
    """
    TV-like prior on face normals over adjacent faces.
    Encourages piecewise-planar surfaces while preserving sharp edges.
    """
    if face_adj_pairs is None or face_adj_pairs.shape[0] == 0:
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    n_unit, _ = compute_face_normals_and_double_area(vertices, faces)
    p = face_adj_pairs
    cos_ij = (n_unit[p[:, 0]] * n_unit[p[:, 1]]).sum(dim=1).clamp(-1.0, 1.0)
    d = 1.0 - cos_ij

    delta_val = float(max(1e-8, delta))
    tv = torch.sqrt(d * d + delta_val * delta_val) - delta_val

    # Unweighted total variance over adjacent face normals, scaled by face count.
    face_count = max(1, int(faces.shape[0]))
    return tv.sum() / float(face_count)


def _geometry_face_edge_uniform_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage each triangle to have similar edge lengths.
    Scale-invariant form: penalize per-face edge-length variance normalized by
    the triangle's mean edge length.
    """
    if faces is None or faces.numel() == 0:
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    l01 = (v0 - v1).norm(dim=1)
    l12 = (v1 - v2).norm(dim=1)
    l20 = (v2 - v0).norm(dim=1)
    edges = torch.stack([l01, l12, l20], dim=1)

    mean_len = edges.mean(dim=1, keepdim=True).clamp(min=float(max(1e-12, eps)))
    rel = edges / mean_len
    return (rel - 1.0).pow(2).mean()


# ---------------------------------------------------------------------------
# Photometric loss
# ---------------------------------------------------------------------------

class PhotometricLoss(nn.Module):
    """
    Combined L1 + SSIM photometric loss.
    Both pred and target should be in [0, 1] (post-ISP LDR space).
    """

    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        ssim_backend: str = "auto",
    ):
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.ssim_backend = ssim_backend

    def _compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backend = self.ssim_backend
        if backend == "msssim":
            return _ssim_loss_msssim(pred, target, mask=mask)
        if backend == "native":
            return _ssim_loss_native(pred, target, mask=mask)
        # auto: prefer accelerated backend when installed.
        if msssim_ssim is not None:
            return _ssim_loss_msssim(pred, target, mask=mask)
        return _ssim_loss_native(pred, target, mask=mask)

    def forward(
        self,
        pred:   torch.Tensor,   # (H, W, 3) predicted LDR image
        target: torch.Tensor,   # (H, W, 3) ground-truth photograph
        mask:   Optional[torch.Tensor] = None,  # (H, W) bool; None = all pixels
    ) -> torch.Tensor:
        if mask is not None:
            valid = mask.bool()
            if not valid.any():
                return torch.zeros((), device=pred.device, dtype=pred.dtype)
            pred_m   = pred[valid]
            target_m = target[valid]
        else:
            pred_m, target_m = pred, target

        l1   = F.l1_loss(pred_m, target_m)
        ssim = self._compute_ssim(pred, target, mask=mask)

        return self.l1_weight * l1 + self.ssim_weight * ssim


# ---------------------------------------------------------------------------
# Per-camera ISP regularisation
# ---------------------------------------------------------------------------

class PPISPRegLoss(nn.Module):
    """
    Soft L2 regulariser to keep PPISP params near identity values.

    Prevents degenerate solutions (e.g. all exposure → 0, all contrast → ∞).
    Weights are typically much smaller than photometric loss.
    """

    def __init__(
        self,
        exposure_weight:   float = 0.01,
        gamma_weight:      float = 0.01,
        wb_weight:         float = 0.01,
        brightness_weight: float = 0.1,
        contrast_weight:   float = 0.01,
        vignette_weight:   float = 0.01,
    ):
        super().__init__()
        self.w_exp  = exposure_weight
        self.w_gam  = gamma_weight
        self.w_wb   = wb_weight
        self.w_bri  = brightness_weight
        self.w_con  = contrast_weight
        self.w_vig  = vignette_weight

    def forward(self, ppisp: "PPISPParams") -> torch.Tensor:
        loss = torch.tensor(0.0, device=ppisp.log_exposure.device)

        # exposure → 1  (log → 0)
        loss = loss + self.w_exp * ppisp.log_exposure.pow(2).mean()

        # gamma → 2.2  (softplus(raw) + 0.5 = 2.2  →  raw ≈ 1.67)
        loss = loss + self.w_gam * (ppisp.gamma() - 2.2).pow(2).mean()

        # wb → 1  (log → 0)
        loss = loss + self.w_wb  * ppisp.log_wb.pow(2).mean()

        # brightness → 0
        loss = loss + self.w_bri * ppisp.brightness.pow(2).mean()

        # contrast → 1  (log → 0)
        loss = loss + self.w_con * ppisp.log_contrast.pow(2).mean()

        # vignette → near-0  (log_k → -4 ≈ k≈0.018)
        if ppisp.learn_vignette:
            loss = loss + self.w_vig * (ppisp.log_vignette_k + 4.0).pow(2).mean()

        return loss


# ---------------------------------------------------------------------------
# Combined loss aggregator
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Combines all losses with configurable weights.
    """

    def __init__(
        self,
        photo_weight:      float = 1.0,
        ppisp_reg_weight:  float = 1e-2,
        geom_normal_tv_weight: float = 0.0,
        geom_normal_tv_delta: float = 1e-3,
        geom_edge_uniform_weight: float = 0.0,
        geom_edge_uniform_eps: float = 1e-8,
        l1_weight:         float = 0.8,
        ssim_weight:       float = 0.2,
        ssim_backend:      str = "auto",
    ):
        super().__init__()
        self.w_photo    = photo_weight
        self.w_ppisp    = ppisp_reg_weight
        self.w_geom_tv  = geom_normal_tv_weight
        self.geom_tv_delta = geom_normal_tv_delta
        self.w_geom_edge_uniform = geom_edge_uniform_weight
        self.geom_edge_uniform_eps = geom_edge_uniform_eps

        self.photo_loss   = PhotometricLoss(
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            ssim_backend=ssim_backend,
        )
        self.ppisp_reg    = PPISPRegLoss()

    def forward(
        self,
        pred:    torch.Tensor,
        target:  torch.Tensor,
        ppisp:   "PPISPParams",
        mask:    Optional[torch.Tensor] = None,
        learn_geometry: bool = False,
        train_geometry: bool = False,
        vertices: Optional[torch.Tensor] = None,
        faces: Optional[torch.Tensor] = None,
        face_adj_pairs: Optional[torch.Tensor] = None,
        face_adj_edges: Optional[torch.Tensor] = None,
        base_face_pair_weight: Optional[torch.Tensor] = None,
        weld_index: Optional[torch.Tensor] = None,
        num_weld_groups: Optional[int] = None,
        weld_group_inv_counts: Optional[torch.Tensor] = None,
    ) -> dict:
        photo = self.photo_loss(pred, target, mask)
        p_reg = self.ppisp_reg(ppisp)

        if learn_geometry and train_geometry and vertices is not None and faces is not None:
            geom_normal_tv = _geometry_normal_tv_loss(
                vertices=vertices,
                faces=faces,
                face_adj_pairs=face_adj_pairs,
                face_adj_edges=face_adj_edges,
                base_face_pair_weight=base_face_pair_weight,
                weld_index=weld_index,
                num_weld_groups=num_weld_groups,
                weld_group_inv_counts=weld_group_inv_counts,
                delta=self.geom_tv_delta,
            )
            geom_edge_uniform = _geometry_face_edge_uniform_loss(
                vertices=vertices,
                faces=faces,
                eps=self.geom_edge_uniform_eps,
            )
        else:
            geom_normal_tv = torch.zeros((), device=pred.device, dtype=pred.dtype)
            geom_edge_uniform = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (self.w_photo   * photo
               + self.w_ppisp   * p_reg
               + self.w_geom_tv * geom_normal_tv
               + self.w_geom_edge_uniform * geom_edge_uniform)

        return {
            "total":     total,
            "photo":     photo.detach(),
            "ppisp_reg": p_reg.detach(),
            "geom_normal_tv": geom_normal_tv.detach(),
            "geom_edge_uniform": geom_edge_uniform.detach(),
        }
